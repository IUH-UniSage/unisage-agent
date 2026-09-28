-- Attempt-level reservation against every enabled PROVIDER budget (up to 2: one
-- DAILY, one MONTHLY), atomically. Called before every provider call attempt (each
-- node, each failover retry).
--
-- KEYS: 3 per scope-period pair, in order: reserved_key, committed_key, inflight_key.
-- When N = 0 (no enabled PROVIDER budget at all for this provider), still writes the
-- field with amount 0 so release_provider has a committed_key to commit the real cost
-- to later - in that case ARGV[10] (the fallback committed_key) is used instead of a
-- KEYS entry.
--
-- ARGV[1] = requestId
-- ARGV[2] = seq (this attempt's line sequence number)
-- ARGV[3] = estimate (micro-USD integer)
-- ARGV[4] = inflight_ttl_seconds
-- ARGV[5] = resv_hash_ttl_seconds
-- ARGV[6] = resv_hash_key (budget:resv:{requestId})
-- ARGV[7] = scope_count (N)
-- ARGV[8 + 4*(i-1) .. 8 + 4*(i-1) + 3] for i = 1..N, one quadruple per scope-period pair:
--   action ("ALERT" | "THROTTLE" | "BLOCK"), limit (micro-USD), throttle_cap, period_ttl_seconds
-- ARGV[last] = fallback_committed_key - the DAILY-period committed key for this provider,
--   always passed, used only when N = 0 so an unconfigured provider's real cost still has
--   somewhere to land for reporting/reconciliation
--
-- Returns "OK" | "DENY_EXCEEDED" | "DENY_THROTTLED". Nothing is written on denial - the
-- FIRST scope-period pair that denies short-circuits.

local n = tonumber(ARGV[7])
local estimate = tonumber(ARGV[3])

local function scope_args(i)
    local base = 8 + 4 * (i - 1)
    return ARGV[base], tonumber(ARGV[base + 1]), tonumber(ARGV[base + 2]), ARGV[base + 3]
end

local function scope_keys(i)
    local base = 3 * (i - 1)
    return KEYS[base + 1], KEYS[base + 2], KEYS[base + 3]
end

for i = 1, n do
    local action, limit, throttle_cap, _period_ttl = scope_args(i)
    local reserved_key, committed_key, inflight_key = scope_keys(i)

    local reserved = tonumber(redis.call("GET", reserved_key) or "0")
    local committed = tonumber(redis.call("GET", committed_key) or "0")
    local total = reserved + committed

    if action == "BLOCK" then
        if total + estimate > limit then
            return "DENY_EXCEEDED"
        end
    elseif action == "THROTTLE" then
        if total >= limit then
            local inflight = tonumber(redis.call("GET", inflight_key) or "0")
            if inflight >= throttle_cap then
                return "DENY_THROTTLED"
            end
        end
    end
end

local applied_reserved = {}
local applied_inflight = {}
local applied_committed = {}
local seen_inflight = {}

for i = 1, n do
    local _action, _limit, _throttle_cap, period_ttl = scope_args(i)
    local reserved_key, committed_key, inflight_key = scope_keys(i)

    redis.call("INCRBY", reserved_key, estimate)
    redis.call("EXPIRE", reserved_key, period_ttl)

    -- inflight_key depends only on scope, not period - dedupe so a scope with both a
    -- DAILY and a MONTHLY budget only increments its shared inflight counter once.
    if not seen_inflight[inflight_key] then
        seen_inflight[inflight_key] = true
        redis.call("INCR", inflight_key)
        redis.call("EXPIRE", inflight_key, ARGV[4])
        table.insert(applied_inflight, inflight_key)
    end

    table.insert(applied_reserved, reserved_key)
    table.insert(applied_committed, committed_key)
end

if n == 0 then
    -- No enabled PROVIDER budget - amount stays 0, but a committed_key must still be
    -- recorded (as a 1-element list, matching the shape release_provider expects) so
    -- release_provider has somewhere to commit the real cost.
    table.insert(applied_committed, ARGV[8 + 4 * n])
end

local field_value = cjson.encode({
    amount = (n > 0) and estimate or 0,
    reserved = applied_reserved,
    inflight = applied_inflight,
    committed = applied_committed,
})
redis.call("HSET", ARGV[6], "p:" .. ARGV[2], field_value)
redis.call("EXPIRE", ARGV[6], ARGV[5])

return "OK"

-- Request-level budget reservation against every enabled SYSTEM/PURPOSE budget,
-- atomically. Called once at the start of a Chat/Extraction/Embedding request, before
-- any provider is called.
--
-- A scope (SYSTEM, or PURPOSE:<name>) can have UP TO 2 simultaneously enabled budgets -
-- one DAILY, one MONTHLY (Java's unique index is per (scope[,ref], period), not per
-- scope alone) - so this script checks/reserves a VARIABLE number of scope-period pairs,
-- not a fixed 2.
--
-- KEYS: 3 per scope-period pair, in order: reserved_key, committed_key, inflight_key.
-- ARGV[1] = requestId
-- ARGV[2] = estimate (micro-USD integer)
-- ARGV[3] = inflight_ttl_seconds (fixed safety TTL, refreshed on every increment)
-- ARGV[4] = resv_hash_ttl_seconds
-- ARGV[5] = now (unix timestamp seconds)
-- ARGV[6] = expiry_zset_key (budget:resv:expiry)
-- ARGV[7] = resv_hash_key (budget:resv:{requestId})
-- ARGV[8] = scope_count (N)
-- ARGV[9 + 4*(i-1) .. 9 + 4*(i-1) + 3] for i = 1..N, one quadruple per scope-period pair:
--   action ("ALERT" | "THROTTLE" | "BLOCK"), limit (micro-USD), throttle_cap, period_ttl_seconds
--
-- Returns "OK" | "REJECT_EXCEEDED" | "REJECT_THROTTLED". Nothing is written on rejection -
-- the FIRST scope-period pair that rejects short-circuits the whole reservation.

local n = tonumber(ARGV[8])

local function scope_args(i)
    local base = 9 + 4 * (i - 1)
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
    local estimate = tonumber(ARGV[2])

    if action == "BLOCK" then
        if total + estimate > limit then
            return "REJECT_EXCEEDED"
        end
    elseif action == "THROTTLE" then
        if total >= limit then
            local inflight = tonumber(redis.call("GET", inflight_key) or "0")
            if inflight >= throttle_cap then
                return "REJECT_THROTTLED"
            end
        end
    end
end

local estimate = tonumber(ARGV[2])
local applied_reserved = {}
local applied_inflight = {}
local applied_committed = {}
local seen_inflight = {}

for i = 1, n do
    local _action, _limit, _throttle_cap, period_ttl = scope_args(i)
    local reserved_key, committed_key, inflight_key = scope_keys(i)

    redis.call("INCRBY", reserved_key, estimate)
    redis.call("EXPIRE", reserved_key, period_ttl)

    -- inflight_key depends only on scope, not period - a scope with BOTH a DAILY and
    -- a MONTHLY budget shares one inflight key across both pairs, so it must only be
    -- incremented once per call, not once per pair.
    if not seen_inflight[inflight_key] then
        seen_inflight[inflight_key] = true
        redis.call("INCR", inflight_key)
        redis.call("EXPIRE", inflight_key, ARGV[3])
        table.insert(applied_inflight, inflight_key)
    end

    table.insert(applied_reserved, reserved_key)
    table.insert(applied_committed, committed_key)
end

local field_value = cjson.encode({
    amount = estimate,
    reserved = applied_reserved,
    inflight = applied_inflight,
    committed = applied_committed,
})
redis.call("HSET", ARGV[7], "req", field_value)
redis.call("EXPIRE", ARGV[7], ARGV[4])
redis.call("ZADD", ARGV[6], tonumber(ARGV[5]) + tonumber(ARGV[4]), ARGV[1])

return "OK"

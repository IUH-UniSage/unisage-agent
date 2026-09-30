-- Called once at the very end of a request (success, error, or client disconnect),
-- always via UsageRecorder.close()'s try/finally. Releases every "p:*" field still
-- hanging around (the code forgot to call release_provider for it - e.g. an attempt
-- that errored before reaching that call), releases the "req" reservation, and commits
-- the request's real total cost to the SYSTEM/PURPOSE scopes that were actually
-- reserved. Idempotent via a settled marker key.
--
-- ARGV[1] = requestId (unused directly, kept for readability/log correlation - the
--           script only ever touches keys given explicitly below)
-- ARGV[2] = resv_hash_key (budget:resv:{requestId})
-- ARGV[3] = settled_marker_key (budget:settled:{requestId})
-- ARGV[4] = settled_marker_ttl_seconds
-- ARGV[5] = actual_total (micro-USD integer - the request's real total cost, PRICED
--           lines' costUsd + UNPRICED lines' estimatedCostUsd, FREE excluded; the
--           caller computes this the same way Java's period-totals query does)
--
-- Returns "OK" always (including the already-settled no-op case).

if redis.call("EXISTS", ARGV[3]) == 1 then
    return "OK"
end

local all_fields = redis.call("HGETALL", ARGV[2])
for i = 1, #all_fields, 2 do
    local field_name = all_fields[i]
    local raw = all_fields[i + 1]
    if field_name:sub(1, 2) == "p:" then
        local data = cjson.decode(raw)
        for _, key in ipairs(data.reserved) do
            redis.call("DECRBY", key, data.amount)
        end
        for _, key in ipairs(data.inflight) do
            redis.call("DECR", key)
        end
        -- The real cost of this specific attempt is unknown here (only the request's
        -- grand total is passed in below) - committing 0 for it here is conservative:
        -- the request-level commit below still adds the true total once, just not
        -- attributed per-attempt. This only happens when code forgot to call
        -- release_provider for an attempt, which release_provider's own docstring
        -- calls out as the "belt" this settle sweep is the "suspenders" for.
    end
end

local req_raw = redis.call("HGET", ARGV[2], "req")
if req_raw then
    local req_data = cjson.decode(req_raw)
    for _, key in ipairs(req_data.reserved) do
        redis.call("DECRBY", key, req_data.amount)
    end
    for _, key in ipairs(req_data.inflight) do
        redis.call("DECR", key)
    end
    for _, key in ipairs(req_data.committed) do
        redis.call("INCRBY", key, tonumber(ARGV[5]))
    end
end

redis.call("DEL", ARGV[2])
redis.call("SET", ARGV[3], "1", "EX", ARGV[4])

return "OK"

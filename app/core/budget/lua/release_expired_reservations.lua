-- Sweeps `budget:resv:expiry` for requestIds whose reservation TTL has passed
-- without ever being settled (a crashed/killed process, or a bug that skipped
-- UsageRecorder.close()). Releases every "req" and leftover "p:*" field the same
-- way settle_request does, EXCEPT it never commits anything - an expired
-- reservation's real cost is unknown, so crediting it to `committed` would either
-- double count (if it eventually does get recorded) or invent spend that never
-- happened. This is purely "give the reserved/inflight capacity back", not a
-- cost record.
--
-- ARGV[1] = now (unix timestamp seconds)
-- ARGV[2] = expiry_zset_key (budget:resv:expiry)
-- ARGV[3] = max_batch (caps work per run so one call never blocks Redis for long)
--
-- Returns the list of requestIds released this run.

local expired = redis.call("ZRANGEBYSCORE", ARGV[2], "-inf", ARGV[1], "LIMIT", 0, tonumber(ARGV[3]))
local released = {}

for _, request_id in ipairs(expired) do
    local hash_key = "budget:resv:" .. request_id
    local all_fields = redis.call("HGETALL", hash_key)

    for i = 1, #all_fields, 2 do
        local raw = all_fields[i + 1]
        local data = cjson.decode(raw)
        for _, key in ipairs(data.reserved) do
            redis.call("DECRBY", key, data.amount)
        end
        for _, key in ipairs(data.inflight) do
            redis.call("DECR", key)
        end
    end

    redis.call("DEL", hash_key)
    redis.call("ZREM", ARGV[2], request_id)
    table.insert(released, request_id)
end

return released

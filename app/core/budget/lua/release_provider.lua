-- Called right when one provider attempt ends (success, error, or cancelled stream),
-- releasing what acquire_provider reserved and committing the attempt's real cost.
-- Idempotent: a field that's already gone (already released) is a no-op, not an
-- error - so a retried release/settle call never double-subtracts or double-commits.
--
-- ARGV[1] = requestId (unused directly - kept for log correlation only)
-- ARGV[2] = seq
-- ARGV[3] = actual (micro-USD integer, the real cost of this one line - 0 for FREE/ERROR)
-- ARGV[4] = resv_hash_key (budget:resv:{requestId})
--
-- Returns "OK" always (including the no-op case).

local field_name = "p:" .. ARGV[2]
local raw = redis.call("HGET", ARGV[4], field_name)
if not raw then
    return "OK"
end

local data = cjson.decode(raw)
local actual = tonumber(ARGV[3])

for _, key in ipairs(data.reserved) do
    redis.call("DECRBY", key, data.amount)
end
for _, key in ipairs(data.inflight) do
    redis.call("DECR", key)
end
for _, key in ipairs(data.committed) do
    redis.call("INCRBY", key, actual)
end

redis.call("HDEL", ARGV[4], field_name)

return "OK"

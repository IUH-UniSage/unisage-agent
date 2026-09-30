-- Atomically reads the outbox+processing pending count and the current
-- `committed` counter for one scope-period key, so the caller can decide
-- whether it is safe to reconcile against Java's DB total this run (a
-- non-empty outbox means some usage logs Java hasn't seen yet, so its total
-- would be stale).
--
-- KEYS[1] = outbox_key
-- KEYS[2] = processing_key
-- KEYS[3] = committed_key
--
-- Returns {pending_count, committed_value}.

local pending = redis.call("LLEN", KEYS[1]) + redis.call("LLEN", KEYS[2])
local committed = tonumber(redis.call("GET", KEYS[3]) or "0")
return { pending, committed }

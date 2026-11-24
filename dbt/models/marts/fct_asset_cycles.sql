-- Labelled cycles: remaining useful life (RUL) per asset-cycle.
-- train: engine runs to failure, so RUL = last cycle - cycle.
-- test:  trajectory is truncated, so RUL = true RUL at last cycle + (last cycle - cycle).
with c as (
    select *, max(cycle) over (partition by asset_id) as last_cycle
    from {{ ref('stg_cycles') }}
)

select
    c.*,
    case when c.split = 'train' then c.last_cycle - c.cycle
         else t.rul_at_last_cycle + (c.last_cycle - c.cycle) end                    as rul,
    least(case when c.split = 'train' then c.last_cycle - c.cycle
               else t.rul_at_last_cycle + (c.last_cycle - c.cycle) end,
          {{ var('rul_cap') }})                                                      as rul_capped,
    c.cycle = c.last_cycle                                                           as is_last_cycle
from c
left join {{ ref('stg_rul_truth') }} t on c.asset_id = t.asset_id

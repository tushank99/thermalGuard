-- One row per asset per operating cycle, train and test unioned, with typed columns and lineage.
with unioned as (
    select *, 'train' as split from {{ source('raw', 'cmapss_train') }}
    union all
    select *, 'test' as split from {{ source('raw', 'cmapss_test') }}
)

select
    subset || '_' || split || '_' || lpad(cast(unit_id as varchar), 3, '0')          as asset_id,
    subset || '_' || split || '_' || lpad(cast(unit_id as varchar), 3, '0')
        || '_' || lpad(cast(cycle as varchar), 3, '0')                                 as cycle_key,
    subset,
    split,
    cast(unit_id as integer)                                                           as unit_id,
    cast(cycle as integer)                                                             as cycle,
    {% for i in range(1, 4) %}cast(op_setting_{{ i }} as double) as op_setting_{{ i }},
    {% endfor %}
    {% for i in range(1, 22) %}cast(sensor_{{ i }} as double) as sensor_{{ i }},
    {% endfor %}
    batch_id,
    loaded_at
from unioned

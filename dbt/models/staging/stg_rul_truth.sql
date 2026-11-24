select
    subset || '_test_' || lpad(cast(unit_id as varchar), 3, '0') as asset_id,
    cast(rul_true as integer)                                    as rul_at_last_cycle
from {{ source('raw', 'cmapss_rul') }}

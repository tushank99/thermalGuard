-- Model-ready features: raw informative sensors + rolling mean/std + slope proxy over the last N cycles.
{% set w = var('rolling_window') - 1 %}
select
    asset_id, cycle_key, subset, split, unit_id, cycle, is_last_cycle, rul, rul_capped,
    op_setting_1, op_setting_2, op_setting_3,
    {% for s in var('informative_sensors') %}
    sensor_{{ s }},
    avg(sensor_{{ s }}) over (partition by asset_id order by cycle rows between {{ w }} preceding and current row)
        as sensor_{{ s }}_mean_w,
    coalesce(stddev_samp(sensor_{{ s }}) over (partition by asset_id order by cycle rows between {{ w }} preceding and current row), 0)
        as sensor_{{ s }}_std_w,
    sensor_{{ s }} - first_value(sensor_{{ s }}) over (partition by asset_id order by cycle rows between {{ w }} preceding and current row)
        as sensor_{{ s }}_delta_w{% if not loop.last %},{% endif %}
    {% endfor %}
from {{ ref('fct_asset_cycles') }}

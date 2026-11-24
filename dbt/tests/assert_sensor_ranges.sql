-- Physical sanity bounds for all 21 sensors: no nulls, no negatives, no absurd spikes.
-- Any returned row fails the build and blocks training on this batch.
{% for i in range(1, 22) %}
select cycle_key, 'sensor_{{ i }}' as sensor, sensor_{{ i }} as value
from {{ ref('stg_cycles') }}
where sensor_{{ i }} is null or sensor_{{ i }} < 0 or sensor_{{ i }} > 10000
{% if not loop.last %}union all{% endif %}
{% endfor %}

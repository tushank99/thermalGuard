{# Range check: fails on any non-null value outside [min_value, max_value]. #}
{% test in_range(model, column_name, min_value, max_value) %}
select {{ column_name }} as bad_value
from {{ model }}
where {{ column_name }} < {{ min_value }} or {{ column_name }} > {{ max_value }}
{% endtest %}


{# Schema check: fails with one row per expected column missing from the relation. Cross-warehouse. #}
{% test has_columns(model, columns) %}
{%- set actual = [] -%}
{%- if execute -%}
  {%- for c in adapter.get_columns_in_relation(model) -%}{%- do actual.append(c.name | lower) -%}{%- endfor -%}
{%- endif -%}
{%- set missing = [] -%}
{%- for c in columns -%}{%- if execute and (c | lower) not in actual -%}{%- do missing.append(c) -%}{%- endif -%}{%- endfor -%}
{% if missing | length == 0 %}
select 1 as missing_column where 1 = 0
{% else %}
{% for m in missing %}select '{{ m }}' as missing_column{% if not loop.last %} union all {% endif %}{% endfor %}
{% endif %}
{% endtest %}

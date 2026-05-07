{{ fullname.split('.')[-1] | escape | underline}}

.. automodule:: {{ fullname }}

{% set ns = namespace(subpackages=[], submodules=[]) %}
{% for item in modules %}
{% set qualified = fullname + '.' + item %}
{% if is_subpackage(qualified) %}
{% set ns.subpackages = ns.subpackages + [item] %}
{% else %}
{% set ns.submodules = ns.submodules + [item] %}
{% endif %}
{% endfor %}
{% if ns.subpackages or ns.submodules %}
.. toctree::
   :hidden:
   :maxdepth: 2

{% for item in ns.subpackages %}
   {{ fullname }}.{{ item }}
{%- endfor %}
{% for item in ns.submodules %}
   {{ fullname }}.{{ item }}
{%- endfor %}
{% endif %}
{% if ns.subpackages %}
Subpackages
-----------

.. autosummary::
   :toctree:
   :recursive:
{% for item in ns.subpackages %}
   {{ item }}
{%- endfor %}
{% endif %}
{% if ns.submodules %}
Modules
-------

.. autosummary::
   :toctree:
   :recursive:
{% for item in ns.submodules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% if attributes %}
Attributes
----------

.. autosummary::
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% if functions %}
Functions
---------

.. autosummary::
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% if classes %}
Classes
-------

.. autosummary::
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% if exceptions %}
Exceptions
----------

.. autosummary::
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}

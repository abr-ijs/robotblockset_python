{{ fullname.split('.')[-1] | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :member-order: bysource
   :show-inheritance:

.. rubric:: Constructor

.. automethod:: {{ fullname }}.__init__

{% if attributes %}
.. rubric:: Attributes

.. autosummary::
{% for item in attributes %}
   ~{{ fullname }}.{{ item }}
{%- endfor %}
{% endif %}

{% if methods %}
.. rubric:: Methods

.. autosummary::
{% for item in methods %}
{% if item != '__init__' %}
   ~{{ fullname }}.{{ item }}
{% endif %}
{%- endfor %}
{% endif %}

---
layout: page
title: "游记"
description: "行千里路，读万卷书"
header-img: "img/blue.jpg"
---


<ul class="listing">
{% for tag in site.tags %}
	{% if tag[0] == 'travel' %}
  <h3 class="listing-seperator" id="{{ tag[0] }}">我的游记</h3>
{% for post in tag[1] %}
  <li class="listing-item">
  <time datetime="{{ post.date | date:"%Y-%m-%d" }}">{{ post.date | date:"%Y-%m-%d" }}</time>
  <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
  </li>
{% endfor %}
{% endif %}
{% endfor %}
</ul>







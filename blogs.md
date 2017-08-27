---
layout: page
title: "技术博客"
description: "学无止境，重在点滴"
header-img: "img/orange.jpg"
---


<ul class="listing">
{% for tag in site.tags %}
	{% if tag[0] == 'tech' %}
  <h3 class="listing-seperator" id="{{ tag[0] }}">我的技术博客</h3>
{% for post in tag[1] %}
  <li class="listing-item">
  <time datetime="{{ post.date | date:"%Y-%m-%d" }}">{{ post.date | date:"%Y-%m-%d" }}</time>
  <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
  </li>
{% endfor %}
{% endif %}
{% endfor %}
</ul>
---
layout: post
title: I built a Baby Care web app using JHipster (open source from now on 🥰)
date: 2022-04-23 00:00:00 -0300
tags: jhipster
image: img/postbanners/cover-built-baby-care-web-app.jpeg
draft: true
youtubeId1: JI0BiPdHV2E
youtubeId2: zjsSHJaPgAk
youtubeId3: Oo9Umj4COj0
youtubeId4: gGAwglrA0RM

---

![cover image](https://renanfranca.github.io/img/postbanners/cover-built-baby-care-web-app.jpeg)

My baby girl Marília was born on December 14th/2021, so I decided to build a web app before she was born to help me to track her sleep hours, favorite nap place, her humor, breastfeeding and more. Her nickname is Mamazinha, so the web app is called Mamazinha Baby Care.

## Features

- Monitor her nap sleep hours

![new nap screen](https://renanfranca.github.io/img/built-baby-care-web-app/new_nap_screen.png)
<figcaption>Register my baby girl nap</figcaption>

![home screen sleep](https://renanfranca.github.io/img/built-baby-care-web-app/home_screen_sleep_progress.png) 
<figcaption>Monitor her sleep hours from last week versus this week. The goal is at least 16 hours/day after summing up every nap</figcaption>

{% include youtubePlayer.html id=page.youtubeId1 %}

- Breast/formula feeding feature
- {% include youtubePlayer.html id=page.youtubeId2 %}

- Favorite nap place
- The mood when weak up and during the day
- Track her size
- Track her weight

![baby profile detail](https://renanfranca.github.io/img/built-baby-care-web-app/baby_profile_detail.png?cache=2313)
<figcaption>Baby profile detail view</figcaption>

## Technical Detail

![Mamazinha jdl file](https://renanfranca.github.io/img/built-baby-care-web-app/mamazinha_jdl_graphic.png)
<figcaption>Mamazinha Baby Care JDL diagram view</figcaption>

So I used [the Gateway with JHipster Registry architecture](https://www.jhipster.tech/api-gateway/) and create a microservice called 'baby' where I implemented the backend features described at the beginning. The gateway implements the front end, the user management, authentication, proxy all requests to the microservices and [it is reactive for optimized performance](https://developer.okta.com/blog/2021/01/20/reactive-java-microservices). Finally, there are two PostGres databases, one used by baby microservice and the other used by the gateway.

Here is the [JDL file](https://www.jhipster.tech/jdl/intro) created to build this project (I explain each line with comments);

<script src="https://gist.github.com/renanfranca/e473ac73e6493357d1ee60699b63101f.js"></script>

After running the Jhipster generator from the jdl file, two projects were created:
 - [renanfranca/mamazinha-gateway](https://github.com/renanfranca/mamazinha-gateway)
 - [renanfranca/mamazinha-baby](https://github.com/renanfranca/mamazinha-baby)

An overview of what I have to change to make my project works as I want on the gateway:

- Add the UserId inside the [JWT token by default it comes with username and role information](https://www.jhipster.tech/security/)
- [Force the https redirect](https://www.jhipster.tech/security/#https), but only when using the prod profile by configuring the application.yml
- Add graphic to the home screen and implement an intuitive step by step usability

{% include youtubePlayer.html id=page.youtubeId3 %}

- Change the radion button by using [font-awesome smile](https://fontawesome.com/v5.15/icons?d=gallery&p=2&q=smile) when registering the baby humor
- Change the baby profile detail interface screen to have everything I need to know about my baby girl

An overview of what I have to change to make my project works as I want on the baby microservice:

- I created a bunch of endpoints
    1. today-sum-naps-in-hours-by-baby-profile
    2. lastweek-currentweek-sum-naps-in-hours-eachday-by-baby-profile
    3. today-average-nap-humor-by-baby-profile
    4. favorite-nap-place-from-last-days-by-baby-profile
    5. latest-height-by-baby-profilelatest-weight-by-baby-profile
    6. lastweek-currentweek-average-breast-feeds-in-hours-eachday-by-baby-profile/{id}
    7. incomplete-breast-feeds-by-baby-profile/{id}
    8. today-breast-feeds-by-baby-profile/{id}
    
## Run it now on your local machine

Here, I will show you how to run the project on development mode and you are going to have access to all the features I described in my blog post [My 15+ Reasons to use JHipster](https://renanfranca.github.io/2022/03/08/my-reasons-to-use-jhipster.html#my-15-reasons-to-use-jhipster).

### Required

- Install [java](https://adoptopenjdk.net/) version 11
- Install [git](https://git-scm.com/)
- Install [nodejs](https://nodejs.org/en/download/)

### Download the projects and run on dev mode

Your folder struct and the command to execute in each one will be like:

📂mamazinha
- 
```
git clone https://github.com/renanfranca/mamazinha-gateway.git
```
- 
```
git clone https://github.com/renanfranca/mamazinha-baby.git
```
- 
```
 java -jar jhipster-registry-7.1.0.jar --spring.profiles.active=dev --spring.security.user.password=admin --jhipster.registry.password=admin --spring.cloud.config.server.composite.0.type=native --spring.cloud.config.server.composite.0.search-locations=file:./mamazinha-gateway/src/main/docker/central-server-config/localhost-config/
 ```
- 📂 mamazinha-gateway

  - `.\mvnw`
  - `npm start`

- 📂 mamazinha-baby

  - `.\mvnw`

- 📄 jhipster-registry.7.1.0.jar

A short screen record video showing how to make it works 🤓
{% include youtubePlayer.html id=page.youtubeId4 %}

## Benefits from that project

- Me and my wife discovered that our baby needs at least 10 breastfeeds to have a sweet night's sleep 💤 (7 to 8 hours in a row)
![breastfeeding 10 times](https://renanfranca.github.io/img/built-baby-care-web-app/breastfeeding_10_times.jpg)
- We realized that our baby needs to wait at least 1 hour between breastfeeds to full digestion
- Easily knows how many slept hours at the current day
![tracking slept hours 12](https://renanfranca.github.io/img/built-baby-care-web-app/tracking_slept_hours_12.jpg)
- Detect a pattern of her nap time

## Why did I open source it?

- I couldn't afford the cost to host it and make it free to use. So now everyone can use it by host it localhost like I am doing now.😁
- I want to share a full project built with JHipster that could be a startup product (in my opinion). So I could enhanced this post by sharing how JHipster is awesome in a "real" application and not in a hypothetical or sample app.🤓
- I want to have an experience by knowing what it's like to make an open source project and see where is going to lead me. ☺️

## I invite you

to start try out JHipster by [executing the command](https://renanfranca.github.io/2022/03/08/my-reasons-to-use-jhipster.html#generating-your-project-using-jhipster-quick-start-steps) `jhipster jdl jhipster-jdl.jdl` with the [mamazinha-baby-care.jdl](https://gist.github.com/renanfranca/e473ac73e6493357d1ee60699b63101f). Then you will compare what jhipster gave to me as a bootstrap project with the final product I showed in this blog post.
Please, ask me anything in the comment section and I am available on Twitter ([@renan_afranca](https://twitter.com/renan_afranca)) by tag me or DM to support your journey to mastery jhipster.

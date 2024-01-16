---
layout: post
title: Calculating Planet Volumes
image: "/posts/Planets.jpeg"
tags: [Python, Numpy, Planet Volumes]
---

In this post I'm going to take the advantage of Numpy to quickly calculate the volume of all planets in our solar system.  At first glance, it seems just some basic math but Numpy adds a great value as we want to calculate the volume of all planets simultaneously.
Once I did that, I will crank it up and do it for 1000000 planets. Doing this, we can see how fast Numpy can undertake this calculations.

Let's get into it!

---

We have radii of planets as follow:
Mercury: 2439.7 km
Venus: 6051.8 km
Earth: 6371 km
Mars: 3389.7 km
Jupiter: 69911 km
Saturn: 58232	km
Uranus: 25362 km
Neptune: 24622 km

First, let's start by setting up an array that contains radii of planets.

```ruby
radii = np.array([2439.7, 6051.8, 6371, 3389.7, 69911, 58232, 25362, 24622])
```

The volume of a sphere is calculated using the formula: 4/3*pi*r^3.
We could create a loop that iterates through each planet and find the volume of all volumes. But, instead, we will use Numpy to see how it can do the job in one swift motion.

```ruby
Planet_Volumes=4/3*np.pi*radii**3
```
Pretty cool! We are instantly returned the volume of eight planets:
```ruby
print(Planet_Volumes)
>>> [6.08272087e+10 9.28415346e+11 1.08320692e+12 1.63144486e+11 1.43128181e+15 8.27129915e+14 6.83343557e+13 6.25257040e+13]
```
But Let's check a cooler thing. I want to overwrite the radii with a one-dimensional array that will contain radii of 1000000 virtual planets between 0 and 1000 km.

```ruby
radii = np.random.randint(0,1000,1000000)
```

We will repeat the previous procedure to see how long it takes to apply the volume formula to 1000000 values in our array.

```ruby
Planet_Volumes=4/3*np.pi*radii**3
```

Wow! That took no time at all. Fractions of seconds.
That shows how impressive Numpy is with its mathematical calculations.

```ruby
print(Planet_Volumes)
>>> [3.34759213e+09 3.33038143e+08 4.66148014e+08 ... 4.13061327e+08 7.98644794e+06 3.76636713e+08]
```

In summary, we could use other methods to calculate the volumes of planets as well but using Numpy gave us a really clean and fast method.

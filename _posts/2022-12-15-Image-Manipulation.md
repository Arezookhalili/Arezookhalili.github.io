---
layout: post
title: Image Manipulation with Python
image: "/posts/camaro.jpg"
tags: [Python, Image manipulation]
---

In this post I'm going to run through a function in Python that can quickly find all the Prime numbers below a given value.  For example, if I passed the function a value of 100, it would find all the prime numbers below 100!

If you're not sure what a Prime number is, it is a number that can only be divided wholly by itself and one so 7 is a prime number as no other numbers apart from 7 or 1 divide cleanly into it 8 is not a prime number as while eight and one divide into it, so do 2 and 4

Let's get into it!

---

First, let's start by importing all the functionalities that we require. 

```ruby
import numpy as np
from skimage import io 
import matplotlib.pyplot as plt
```

In the next step, we will import the image that we are going to work on. The image will be in the form of a Numpy array as images in a digital sense are just a series of numbers representing the intensity of each pixel. In a color image, we have intensity values for each of the three color channels red, green, and blue (RGB).

```ruby
camaro = io.imread ("camaro.jpg")
print(camaro)
```

Let's take a look at the structure of our image.

```ruby
camaro.shape
```

It has a shape of 1200 rows and 1600 columns of pixels showing that our picture is wider than it is tall. Also, it is a three-dimensional array as it shows a color image. 

Now, let's check our camaro image:

```ruby
plt.imshow(camaro)
plt.show()
```

Here, we want to slice the array to crop our image and keep the car only.

```ruby
cropped=camaro[350:1100,200:1400,:]         
plt.imshow(cropped)
plt.show()
```

Then, we save our image before proceeding to the next step.

```ruby
io.imsave('camaro_cropped.jpg', cropped)
```
![alt text](/img/posts/camaro_cropped.png)

We're going to vertically and horizontally flip our image to get familiar with another toolkit.

```ruby
vertical_flip=camaro[::-1,:,:]
plt.imshow(vertical_flip)
plt.show()

io.imsave('camaro_vertical_flip.jpg', vertical_flip)
```
![alt text](/img/posts/Vertical Flip.png)

```ruby
horizontal_flip=camaro[:,::-1,:]
plt.imshow(horizontal_flip)
plt.show()

io.imsave('camaro_horizontal_flip.jpg', horizontal_flip)
```
![alt text](/img/posts/camaro_horizontal_flip.jpg)

There is a method that shows us our image with different color channels. We extract the red, green, and blue versions of our image by zeroing out the other color channels instead of cropping them. For that, we create an array of zeros which is the same size as the camaro image, and then fill only the desired color channel with the values of the same color from the actual image. The next thing is to ensure that our data type is unit8 which is the type of data we often want when we deal with images in Numpy. 
Let's start with the red version of camaro image:

```ruby
red=np.zeros(camaro.shape, dtype='uint8')    
red[:,:,0]=camaro[:,:,0]                     
plt.imshow(red)
plt.show()
```
![alt text](/img/posts/Red%20Car.png)

Next, a green version of the image will be created:

```ruby
green=np.zeros(camaro.shape, dtype='uint8')    
green[:,:,1]=camaro[:,:,1]                     
plt.imshow(green)
plt.show()
```
![alt text](/img/posts/Green%20Car.png)

The final image would be the blue version:

```ruby
blue=np.zeros(camaro.shape, dtype='uint8')    
blue[:,:,2]=camaro[:,:,2]                     
plt.imshow(blue)
plt.show()
```
![alt text](/img/posts/Blue%20Car.png)

Now, Let's do something funny. We are going to use the stacking function to horizontally stack our single-color images beside each other.

```ruby
camaro_rainbow=np.hstack((red,green,blue))    
plt.imshow(camaro_rainbow)
plt.show()

io.imsave('camaro_rainbow_hstack.jpg', camaro_rainbow)
```
![alt text](/img/posts/camaro_rainbow_hstack.jpg)

We then will follow similar steps to vertically stack our single-color images on top of each other.

```ruby
camaro_rainbow=np.vstack((red,green,blue))    
plt.imshow(camaro_rainbow)
plt.show()

io.imsave('camaro_rainbow_vstack.jpg', camaro_rainbow)
```
![alt text](/img/posts/camaro_rainbow_vstack.jpg)

Now, we know that the very first value in our range is going to be a prime...as there is nothing smaller than it so therefore nothing else could possible divide evenly into it.  As we know it's a prime, let's add it to our list of primes...

```ruby
primes_list.append(prime)
print(primes_list)
>>> [2]
```

Now we're going to do a special trick to check our remaining number_range for non-primes. For the prime number we just checked (in this first case it was the number 2) we want to generate all the multiples of that up to our upper range (in our case, 20).

We're going to again use a set rather than a list, because it allows us some special functionality that we'll use soon, which is the magic of this approach.

```ruby
multiples = set(range(prime*2, n+1, prime))
```

Remember that when created a range the syntax is range(start, stop, step). For the starting point - we don't need our number as that has already been added as a prime, so let's start our range of multiples at 2 * our number as that is the first multiple, in our case, our number is 2 so the first multiple will be 4. If the number we were checking was 3 then the first multiple would be 6 - and so on.

For the stopping point of our range - we specify that we want our range to go up to 20, so we use n+1 to specify that we want 20 to be included.

Now, the **step** is key here.  We want multiples of our number, so we want to increment in steps *of our* number so we can put in **prime** here

Lets have a look at our list of multiples...

```ruby
print(multiples)
>>> {4, 6, 8, 10, 12, 14, 16, 18, 20}
```

The next part is the magic I spoke about earlier, we're using the special set functionality **difference_update** which removes any values from our number range that are multiples of the number we just checked. The reason we're doing this is because if a number is a multiple of anything other than 1 or itself then it is **not a prime number** and can remove it from the list to be checked.

Before we apply the **difference_update**, let's look at our two sets.

```ruby
print(number_range)
>>> {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

print(multiples)
>>> {4, 6, 8, 10, 12, 14, 16, 18, 20}
```

**difference_update** works in a way that will update one set to only include the values that are *different* from those in a second set

To use this, we put our initial set and then apply the difference update with our multiples

```ruby
number_range.difference_update(multiples)
print(number_range)
>>> {3, 5, 7, 9, 11, 13, 15, 17, 19}
```

When we look at our number range now, all values that were also present in the multiples set have been removed as we *know* they were not primes

This is amazing!  We've made a massive reduction to the pool of numbers that need to be tested so this is really efficient. It also means the smallest number in our range *is a prime number* as we know nothing smaller than it divides into it...and this means we can run all that logic again from the top!

Whenever you can run sometime over and over again, a while loop is often a good solution.

Here is the code, with a while loop doing the hard work of updated the number list and extracting primes until the list is empty.

Let's run it for any primes below 1000...

```ruby
n = 1000

# number range to be checked
number_range = set(range(2, n+1))

# empty list to append discovered primes to
primes_list = []

# iterate until list is empty
while number_range:
    prime = number_range.pop()
    primes_list.append(prime)
    multiples = set(range(prime*2, n+1, prime))
    number_range.difference_update(multiples)
```

Let's print the primes_list to have a look at what we found!

```ruby
print(primes_list)
>>> [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
```

Let's now get some interesting stats from our list which we can use to summarise our findings, the number of primes that were found, and the largest prime in the list!

```ruby
prime_count = len(primes_list)
largest_prime = max(primes_list)
print(f"There are {prime_count} prime numbers between 1 and {n}, the largest of which is {largest_prime}")
>>> There are 168 prime numbers between 1 and 1000, the largest of which is 997
```

Amazing!

The next thing to do would be to put it into a neat function, which you can see below:

```ruby
def primes_finder(n):
    
    # number range to be checked
    number_range = set(range(2, n+1))

    # empty list to append discovered primes to
    primes_list = []

    # iterate until list is empty
    while number_range:
        prime = number_range.pop()
        primes_list.append(prime)
        multiples = set(range(prime*2, n+1, prime))
        number_range.difference_update(multiples)
        
    prime_count = len(primes_list)
    largest_prime = max(primes_list)
    print(f"There are {prime_count} prime numbers between 1 and {n}, the largest of which is {largest_prime}")
```

Now we can jut pass the function the upper bound of our search and it will do the rest!

Let's go for something large, say a million...

```ruby
primes_finder(1000000)
>>> There are 78498 prime numbers between 1 and 1000000, the largest of which is 999983
```

That is pretty cool!

I hoped you enjoyed learning about Primes, and one way to search for them using Python.

---

###### Important Note: Using pop() on a Set in Python

In the real world - we would need to make a consideration around the pop() method when used on a Set as in some cases it can be a bit inconsistent.

The pop() method will usually extract the lowest element of a Set. Sets however are, by definition, unordered. The items are stored internally with some order, but this internal order is determined by the hash code of the key (which is what allows retrieval to be so fast). 

This hashing method means that we can't 100% rely on it successfully getting the lowest value. In very rare cases, the hash provides a value that is not the lowest.

Even though here, we're just coding up something fun - it is most definitely a useful thing to note when using Sets and pop() in Python in the future!

The simplest solution to force the minimum value to be used is to replace the line...

```ruby
prime = number_range.pop()
```

...with the lines...

```ruby
prime = min(sorted(number_range))
number_range.remove(prime)
```

...where we firstly force the identification of the lowest number in the number_range into our prime variable, and following that we remove it.

However, because we have to sort the list for each iteration of the loop in order to get the minimum value, it's slightly slower than what we saw with pop()!

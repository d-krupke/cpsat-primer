# Hands-On Large Neighborhood Search

In this chapter, we will develop a Large Neighborhood Search (LNS) algorithm to
learn hands-on what a powerful and flexible framework it is for solving many
types of optimization problems. Assume, we opened a very successful bike repair
shop where people bring in their bikes for repair and we will deliver the
repaired bikes back to the customers using a cargo bike that can store six
bikes. At the beginning, this was very easy to do by hand, but now the shop has
grown and we need to increase the efficiency of our delivery process as there
are now hundreds of bikes to deliver every day for our ten delivery staffs.

The first step is to decide which constraints we should consider and which
objectives we should optimize. For this, it can make sense to think of how you
would intuitively do it for small instances by hand. We obviously want to
minimize the effort and time by the delivery staffs, but just letting the
fastest delivery staff take all the bikes is not a good idea as we want to
balance the workload among the delivery staffs. It could make sense to have a
min max objective where we minimize the maximum time spent by a delivery staff,
with a secondary objective of minimizing the total time spent by all delivery
staffs. However, we notice that the delivery staff quite often changes because
we are relying on students, and on top of that, they work different hours. As
there are quite a few tours to be done as every tour can only deliver a few
bikes, it actually makes sense to completely ignore the assignment of bikes to
delivery staffs and just focus on the tours. As soon as a delivery staff returns
to the shop, they will take the next tour that is ready to go which makes the
optimization problem much easier and also more robust. It won't be absolutely
fair all the time, but there is little chance that we could achieve that anyway.

So, this reduces our problem to just find a set of tours limited to six bikes
each that minimizes the sum of the tour times. To estimate the tour time, we
will rely on the distances as a proxy for the time.

## Exercises for Extension

### The is also a truck but it cannot go everywhere because of parking issues. How would you extend the model to also consider the truck?

### There is a time limit and we may not be able to make all deliveries on the same day. If we have to postpone a delivery, it goes up in priority. How would you extend the model to consider this?

### We have two different types of cargo bikes. One can store six bikes, the other can store eight bikes. How would you extend the model to consider this?

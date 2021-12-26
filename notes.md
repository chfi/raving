


what do i need


## basics

allocating memory
creating buffers
creating images

presenting to swapchain

start & end render pass

submit commands

fences & semaphores????

host & device-local memory

copying data to/from buffers and images

memory barriers


## next steps

first i need to set up the window, surface, and swapchain

then i want to create the simplest possible render graph, consisting of just a clear color

then i should be able to follow the vk guide double buffering chapter

after that it should be simple to load and display meshes

then a material system, and an asset system, i guess

## hmmmm

actually i think i have pretty much all the core state i need??

now it's the actual pipeline state, resource management, etc., that i need to take care of

## is this possible

can i just record each of the command buffers used to complete the render graph,

then the semaphores and how their wait/signal relationships match with the command buffer batches

*then* decide how to distribute them among the queues

will that just *work*, assuming the command recording and other pieces are synced and handled correctly?
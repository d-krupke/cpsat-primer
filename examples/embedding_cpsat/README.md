# Embedding CP-SAT in an application

If you want to embed CP-SAT in your application for a potentially long running optimization, you can try to utilize the callbacks to give users updates over the progress and potentially even interrupt it early.
However, one problem is that this may not be sufficient as the application will only be able to react during the callback.
As the callback is not always called frequently, this may lead in problematic delays making it not feasible for GUIs or APIs.
In alternative is to let the solver run in a separate process and communicate with it using a pipe.
This way, the solver can be interrupted at any time and the application can react immediately.
Python's multiprocessing module provides reasonable simple tools to achieve this.
This example showcases such an approach.
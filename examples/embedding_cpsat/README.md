# Embedding CP-SAT in an Application

If you want to embed CP-SAT in your application for potentially long-running optimization tasks, you can use callbacks to provide users with progress updates and potentially interrupt the process early. However, one issue is that the application can only react during the callback. Since the callback is not always called frequently, this may lead to problematic delays, making it unsuitable for graphical user interfaces (GUIs) or application programming interfaces (APIs).

An alternative is to let the solver **run in a separate process** and communicate with it using a pipe. This approach allows the solver to be interrupted at any time, enabling the application to react immediately. Python's multiprocessing module provides reasonably simple tools to achieve this. The following example showcases such an approach.

## Installation

This demo is a streamlit app showcasing multiprocessing. The simplest way to run it on your computer is to use [uv python package manager](https://github.com/astral-sh/uv).

1. Install uv.

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Run the app with this command.

```bash
uv run streamlit run app.py
```

3. Open your browser and try the app (the default URL is [http://localhost:8501](http://localhost:8501))

## Deployment

The easiest way to deploy this app to the internet for free is to use the [Streamlit community cloud](https://streamlit.io/cloud).

1. Make sure you forked this repository
2. Register to Streamlit cloud using your git account
3. Create a new Streamlit cloud app. Enter the url to your git repository and the path to the `app.py` file : `examples/embedding_cpsat/app.py`
4. You'll be redirected to a website with your app. You can now share this URL with other people.

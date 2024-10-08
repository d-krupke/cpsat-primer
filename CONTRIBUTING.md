# Contributing to the CP-SAT Primer Book

Thank you for your interest in contributing to the CP-SAT Primer book!

- Discussions and issues are managed in the
  [GitHub repository](https://github.com/d-krupke/cpsat-primer/issues).
- Feel free to open a new issue for discussions or email me directly.
- All contributions are welcome, and substantial contributions will be
  recognized with co-authorship.

## Book Code Organization

Each chapter of the book has as a separate Markdown file in the root directory
(e.g., `00_intro.md` for the introduction). If you want to fix typos or add
material, edit these chapters directly in the root directory. Please do not make
edits to the `README.md` file as this file will be automatically generated by
running the `build.py` script.

If you install [pre-commit](https://pre-commit.com/), formatting and building
will be done automatically for every commit. pre-commit can be installed from
pip via `pip install pre-commit` and activated with `pre-commit install` in the
repository.

## Building the Book Locally

To preview the changes that you make to the book:

1. Install `mdbook` using the
   [official mdBook installation guide](https://rust-lang.github.io/mdBook/guide/installation.html).

2. From the root directory, run:

   ```
   python3 build.py
   ```

3. Navigate to the `.mdbook` directory:

   ```
   cd .mdbook
   ```

4. Build the book:

   ```
   mdbook build
   ```

5. To view and automatically update the book in your browser:

   ```
   mdbook watch --open .
   ```

6. After any changes, rerun `python3 build.py` from the root directory to see
   the changes reflected in your browser.

## Submitting a Pull Request

1. Fork the repository to your GitHub account.

2. Make your changes directly to the relevant Markdown files in the root
   directory of your fork.

3. Build and preview your changes locally by following the steps in the
   [Building the Book Locally](#building-the-book-locally) section above.

   ```
   python3 build.py
   cd .mdbook
   mdbook build
   mdbook watch --open
   ```

4. Commit your changes (including changes to the `README.md` file) with a clear
   and descriptive commit message:

   ```
   git commit -am "Add description of your changes"
   ```

5. Push your changes to your forked repository:

   ```
   git push origin main
   ```

6. Go to the original repository on GitHub and click "New pull request".

7. Select your fork and ensure the changes you want to submit are included.

8. Provide a clear title and description for your pull request, explaining the
   changes you've made.

9. Submit the pull request for review.

10. Be prepared to make additional changes if requested by the maintainers.

## Testing

CP-SAT is under rapid development, and its behavior can change from time to
time. Therefore, while it is not mandatory, it is highly appreciated if code
snippets are converted into simple test cases. These test cases do not need to
be complex; even detecting syntax errors when CP-SAT changes is valuable. All
test cases will be automatically run every week against the latest version of
CP-SAT to ensure that the code snippets remain valid.

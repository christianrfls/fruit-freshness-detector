# main.py
# Entry point for the Fruit Freshness Detector.
# The user interface is a Streamlit web app (app.py). Running this file
# starts that app in the browser, so markers can launch the project with
# a single command:
#
#     python main.py
#
# (You can also start it directly with:  streamlit run app.py)

import sys
from streamlit.web import cli as stcli


def main():
    # Hand control to Streamlit, telling it to run our app file.
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()

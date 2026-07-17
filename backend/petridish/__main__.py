"""Run the experiment server with `python -m petridish`."""

import uvicorn


def main() -> None:
    uvicorn.run("petridish.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

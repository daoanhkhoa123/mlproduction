@REM Prepare to push
uv pip compile pyproject.toml -o requirements.txt

@REM Push
git commit -m "message"
git push origin

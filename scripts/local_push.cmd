@REM Prepare to push
uv export --format requirements.txt --no-hashes > requirements.txt


@REM Push
git commit -m "message"
git push origin

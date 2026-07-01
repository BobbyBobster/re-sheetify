# hadolint global ignore=DL3007,DL3008,DL3013,DL3042
FROM trainer-base:latest

COPY sheets /code/sheets
COPY Makefile /code/Makefile

CMD ["make", "symlink_and_save_basic"]

# Submission portal

Backend of https://antadatabase.com/submit/ (`web/submit/`, `web/src/submit/`). It receives the
form and the files, and stores each submission as a dataset directory ready for `CompileDatabase`:

```
data/submissions/<id>/
    submission.json                     the form: contributor, publication, ages, status
    <Dataset>/raw/*.csv|*.tab           the files, as uploaded
    <Dataset>/raw_md.json               metadata, as CompileDatabase reads it
    <Dataset>/original_new_column_names.csv
```

The contributors use the standard column names, so `original_new_column_names.csv` is an identity
mapping, apart from the layer columns of the line files (`IRH1, IRH2...` -> IRH numbers).

The portal also sends the messages of the contact page (`/contact/`) to the maintainer. Their
files (up to 10 files, 500 MB) are not emailed but stored with the message, and only accepted
from a verified email address.

Online, the files go to private buckets (`antadatabase-submissions`, `antadatabase-contact`,
under `<id>/`), and the server only keeps the small records (`submission.json`); locally,
without buckets, everything stays in `data/`.

## Compilation and 3D preview

The worker (`worker.py`, a container of its own) compiles each new submission, one at a time,
with `pipeline.py`:

1. **Checking the files.** The raw files are checked again with the rules of the form
   (`checks.check_raw`), since the API can be called without it. Errors stop here.
2. **Compiling** with `CompileDatabase`, then **assigning the ages** with
   `IndexDatabase(assign_ages=True)`.
3. **Checking the compiled dataset** (`checks.check_compiled`). These checks only give warnings
   for the review:
   - points lost by the compilation;
   - IRHs deeper than the ice thickness (of the dataset, or Bedmap3);
   - IRHs crossing others of different ages (a wrong number or age);
   - spikes;
   - points off the ice.
4. **Building the 3D view** (`anta_database.web.irh`) into `data/previews/<id>/`. The portal
   serves it to whoever has the key of the submission: `/viewer/?preview=<id>&key=<key>`. The
   link is on the receipt of the form and in the receipt email. While the worker runs, the page
   shows its progress, then the warnings of the checks (`web/src/preview.ts`).
5. **Storing** the compiled dataset (h5 files, `raw_md.json`, `AntADatabase.db`, `report.json`)
   in the bucket of its purpose: `antadatabase-compiled` (to add to the database after review)
   or `antadatabase-review` (review only, never redistributed).

The maintainer then gets an email with the report. The contributor only gets one if the
compilation failed (why, when the files were at fault): otherwise the preview page tells them. Each compilation runs in a process of its own, with a time limit, and the container
has a memory limit, so a failing compilation only fails that submission.

The 3D view and the checks need the ice surface and thickness of Bedmap3, as a small file
(84 MB) that `deploy.sh` copies to `data/bedmap3_surface.nc`:

```python
import xarray as xr
ds = xr.open_dataset("bedmap3.nc")[["usurf", "thk"]]
enc = {v: {"dtype": "int16", "_FillValue": -9999, "zlib": True, "complevel": 4, "chunksizes": (500, 500)} for v in ds}
ds.to_netcdf("bedmap3_surface.nc", encoding=enc)
```

## Review: accept or refuse a submission

The review runs on the maintainer's computer with `portal/review.py`, as the command
`anta-review`, from any directory. Make the command once, from the repository:

```bash
chmod +x portal/review.py && ln -s "$PWD/portal/review.py" ~/.local/bin/anta-review
```

It uses the `python` of the active environment, which needs `anta_database[publish,compile]`. The
server has no keys for the public bucket, so a break-in there cannot change the published
database. `review.py` drives the server over SSH (`manage.py`, which sends the emails), and
publishes from a **release copy** of the database (default
`~/data/isochrones/AntADatabase-release`). The tool makes that copy identical to the published
bucket before each change: sync, then a checksum of every file. Your own working copy of the
database is never used.

It needs:
- the AWS profiles `infomaniak` (public bucket) and `antadatabase-portal` (portal buckets);
- SSH access to `api.antadatabase.com`;
- the full `bedmap3.nc`.

The settings are in `~/.config/antadatabase/review.json`.

Once, set up the release copy (downloads the database, about 630 MB):

```bash
anta-review init
```

For each submission, after the email "Compiled: ...":

```bash
anta-review list                 # the submissions and their status
anta-review show <id>            # checks report, contributor, notes, preview link (open it)
anta-review pull <id>            # optional: ./<id>/ with the compiled dataset and the raw files,
                                             # to look at in Python: Database("./<id>")  (--to DIR elsewhere)
```

Then decide.

**Accept:**

```bash
anta-review accept <id>
```

`accept` asks before publishing. It does, in order:
1. Makes the release copy identical to the published database.
2. Adds the dataset to it, and rebuilds the index.
3. Exports the 3D viewer and download page data.
4. Publishes the database and these data.
5. Checks them online.
6. Publishes the raw files to `raw/<Dataset>/` in the public bucket (below).
7. Closes the submission on the server: removes the preview and the compiled copy, keeps the
   raw files, and emails the contributor.
8. Adds the dataset to `book/intro.md` and `book/changelog.md`.

`raw/<Dataset>/` holds, for the record and for later versions of the database:
- `raw/`: the files as the contributor sent them, with all their columns. That includes the
  optional ones the compilation doesn't use yet (`IRH_DEPTH_UNC`, `IRH_TWT`, `TRACE_NUM`,
  `TIME`...) and any unknown ones;
- `raw_md.json` and `original_new_column_names.csv`, to compile them again;
- `provenance.json`: the submission (id, dates, files and their columns, ages, notes), without
  the contributor's name and email.

It sits at the root of the bucket, outside the database prefixes (`v1/`, `v2/`...). There is a
single copy whatever the data schema, and it is in no manifest, so `sync()` and the download
page never fetch it. `accept --replace` replaces it, and `remove` deletes it (the copy in
`antadatabase-submissions` stays).

It can be run again after a failure, and goes on from where it stopped. The viewer and the
download page show the new dataset straight away; the docs need:

```bash
git add book/intro.md book/changelog.md book/_toc.yml && git commit -m "data: add <Dataset>"
./publish_site.sh
```

**Refuse:**

```bash
anta-review reject <id>          # message to the contributor in $EDITOR; all the data is kept
anta-review purge <id>           # later: delete it (list shows refusals older than 90 days)
```

**Review only:**

```bash
anta-review done <id>            # message to the contributor; all its data deleted, as promised
```

**Compile again**, after a fix of the package:

```bash
anta-review retry <id>
```

**Take a published dataset out again**, e.g. one accepted by mistake:

```bash
anta-review remove <Dataset> --reason "..."      # --message "..." to also email its contributor
```

`remove` works like `accept`, the other way round, and asks before publishing:
1. Makes the release copy identical to the published database.
2. Takes the dataset out, and rebuilds the index and the viewer data.
3. Publishes a new database version without it, then checks online that it is gone, and
   deletes its raw files from `raw/<Dataset>/`.
4. If the dataset came through the portal, marks its submission `withdrawn` (the raw files stay
   in `antadatabase-submissions`).
5. Removes it from `book/intro.md` and adds a line to `book/changelog.md`; commit, then
   `./publish_site.sh`.

The next `sync()` of a user removes the dataset from their copy too, and says so. Copies already
downloaded cannot be taken back.

**Test submissions**, sent from `/submit/?dev`, are deleted completely, record included:

```bash
anta-review delete <id>
```

Options of `accept`:
- `--replace`: a corrected version of a published dataset;
- `--db-version`: a version name, instead of the date and time;
- `--test`: publishes to `test/` in the bucket, and leaves the submission and the docs alone.
  Test submissions, sent from `/submit/?dev`, can only be accepted this way.

On the server itself, the same commands are
`cd /opt/portal && docker compose exec worker python -m portal.manage list|show|retry|accept|reject|done|purge|withdraw|delete`.

## Against spam

- Contributors verify their email address with a 6-digit code sent to it (valid 30 minutes, 5
  tries) before they can submit. The verification holds for 24 hours, and submissions are only
  accepted from the verified address.
- Rate limits per address and per IP: verification codes, submissions (10 per day) and
  contact messages.
- A hidden field that only bots fill in, in both forms: their requests are dropped silently.
- Size limits: 100 files, 1 GB per file and 5 GB per submission (`PORTAL_MAX_GB`). The form
  reads the files in pieces, so a 1 GB file takes about 25 s to check in the browser.

ORCID login could replace the email codes later (it needs a registered ORCID API client, with
the portal on its own address).

## Closing the portal

```bash
echo "Submissions are paused until Monday." > portal/data/CLOSED   # close, with a message
rm portal/data/CLOSED                                               # open again
```

While closed, the portal refuses codes, submissions and contact messages, and the submit page
shows the message. No restart is needed.

## Emails

The settings live in `portal/.env` (not in git), see `portal/.env.example`: your address as
`PORTAL_MAINTAINER_EMAIL` (new submissions and contact messages), and the SMTP account sending
the emails (verification codes, confirmations). Without `SMTP_HOST`, emails are only written to
the log. Test the settings with

```bash
portal/.venv/bin/python -m portal.mail     # sends a test email to PORTAL_MAINTAINER_EMAIL
```

To see the emails locally without a real account, run a test mail server and point the portal
at it, without reading `portal/.env`:

```bash
portal/.venv/bin/pip install aiosmtpd
PYTHONUNBUFFERED=1 portal/.venv/bin/python -m aiosmtpd -n -l localhost:1025    # prints the emails
PORTAL_ENV_FILE=/nonexistent SMTP_HOST=localhost SMTP_PORT=1025 SMTP_SECURITY=none portal/.venv/bin/uvicorn portal.app:app --reload --reload-dir portal
```

## Run locally

```bash
python -m venv portal/.venv
portal/.venv/bin/pip install -r portal/requirements.txt
portal/.venv/bin/uvicorn portal.app:app --reload --reload-dir portal --port 8000   # from the repository root
python -m portal.worker   # compiles the submissions: needs anta_database[compile] in that environment

cd web && npm run dev     # http://localhost:5173/submit/, /api is proxied to port 8000
```

In a build of the site, the forms only use the portal when `VITE_PORTAL_URL` is set to its
address; otherwise the submit page says the portal is closed, and the contact page opens an
email instead.

## Test data

```bash
python portal/dev/make_fake_data.py
```

turns the compiled Chung_2023 back into raw files, as dataset Fake_2026, in `portal/dev/fake_data/`:
`per_layer/` (one .tab file per IRH, plus ICE_THK and BASAL_UNIT), `per_line/` (one .csv file per
line, lon/lat), `per_layer_columns/` (ICE_THK and SURF_ELEV in the layer files, with optional and
unknown columns) and `broken/` (mistakes the form should catch). Chung_2023 only has ICE_THK, so
`per_layer/` and `per_line/` also test the warning about the missing reference of the depths. In development, the button
"Fill with the fake submission" of the page fills the fields (data/dev/fake_submission.json); then
drop the files of one of the folders.

A stored submission compiles with the package as any dataset:

```python
from anta_database.compiling import CompileDatabase
CompileDatabase(["portal/data/submissions/<id>/Fake_2026"]).compile()
```

## Online: api.antadatabase.com

The portal runs on its own small server, in its own Infomaniak Public Cloud project
(`antadatabase-portal`, PCP-TEVBM7D), apart from the published data: a break-in on the server
could not touch the public bucket.

| | |
|---|---|
| Server | `antadatabase-portal`: a1-ram2 (1 vCPU, 2 GB), 30 GB disk, Ubuntu 24.04, 195.15.241.223 / 2001:1600:10:100::968 |
| Firewall | security group `portal`: SSH, HTTP, HTTPS only |
| Login | `ssh ubuntu@api.antadatabase.com` (key `anthe-laptop`, no passwords) |
| Buckets | `antadatabase-submissions`, `antadatabase-contact`, `antadatabase-compiled`, `antadatabase-review`: private, with the keys of the project |
| DNS | `api` A and AAAA records to the server |

On the server, `/opt/portal` holds:
- the code, and the `anta_database` package (`package/`, copied by `deploy.sh`);
- `portal.env`: the settings, readable by `ubuntu` only;
- `data/`: the records of the submissions, the verification codes, the previews and Bedmap3.

It runs three containers (`docker-compose.yml`):
- the portal;
- the worker, with 1.4 GB of memory at most (the server also has 2 GB of swap);
- Caddy, which serves the portal over HTTPS with a certificate it gets and renews by itself. The server installs its security updates every night and reboots
when one needs it (`cloud-init.yaml`); the containers start again by themselves.

```bash
portal/deploy.sh                       # update: copy the code and the settings, rebuild, restart
ssh ubuntu@api.antadatabase.com 'cd /opt/portal && docker compose logs --tail 50 portal'   # logs
ssh ubuntu@api.antadatabase.com 'echo "Paused for maintenance." > /opt/portal/data/CLOSED'  # close
```

The settings sent by `deploy.sh` come from `portal/.env` (email) and `portal/.env.server` (keys of
the buckets), both kept out of git. The submit page is open by default. Built with
`PORTAL_PUBLIC=false ./publish_site.sh`, it shows a placeholder instead, and the form only stays
in its development version (`/submit/?dev`). The contact page uses the portal as soon as
`api.antadatabase.com` resolves.

To read the buckets from your computer, with the keys of the portal project:

```bash
aws --profile antadatabase-portal --endpoint-url https://s3.pub1.infomaniak.cloud s3 ls s3://antadatabase-submissions/
aws --profile antadatabase-portal --endpoint-url https://s3.pub1.infomaniak.cloud s3 sync s3://antadatabase-submissions/<id>/ <local dir>
```

The server itself was created once with the OpenStack CLI (project openrc in
`~/.config/openstack/PCU-TEVBM7D-openrc.txt`):

```bash
openstack keypair create --public-key ~/.ssh/id_ed25519.pub anthe-laptop
openstack security group create portal   # + rules: tcp 22, 80, 443 and udp 443, IPv4 and IPv6, and ICMP
openstack server create antadatabase-portal --flavor a1-ram2-disk0 --image "Ubuntu 24.04 LTS Noble Numbat" \
  --boot-from-volume 30 --network ext-net1 --security-group portal --key-name anthe-laptop \
  --user-data portal/cloud-init.yaml --wait
```

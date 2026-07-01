# Setting up the `oci.chi` AWS CLI profile

`research-datasets-chicago` is OCI Object Storage behind an S3-compatible endpoint, not real AWS S3. The AWS CLI needs a profile that points at that endpoint instead of `s3.amazonaws.com`.

## `~/.aws/config`

Add (or confirm) these blocks:

```ini
[profile oci.chi]
region = us-chicago-1
services = oci-s3-compat-chi
request_checksum_calculation = when_required
response_checksum_validation = when_required
s3 =
    max_concurrent_requests = 50
    multipart_chunksize = 16MB
    multipart_threshold = 16MB
    payload_signing_enabled = true

[services oci-s3-compat-chi]
s3 =
    endpoint_url = https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com
```

The `request_checksum_calculation` / `response_checksum_validation` = `when_required` pair matters: newer AWS CLI versions default to always sending checksums, which OCI's S3-compatible endpoint doesn't support and will reject.

## `~/.aws/credentials`

```ini
[oci.chi]
aws_access_key_id = <your OCI Customer Secret Key access key>
aws_secret_access_key = <your OCI Customer Secret Key secret>
```

These come from an OCI Customer Secret Key (Console → Identity → Users → your user → Customer Secret Keys), not your normal OCI API signing key. Never paste the actual values into chat or into this file — copy them directly into `~/.aws/credentials` from the OCI console or from wherever you've already stored them (e.g. a teammate's known-good config, a password manager, or a prior host's copy of this file).

## Verifying it works

```bash
aws configure list --profile oci.chi
aws --profile oci.chi s3 ls s3://research-datasets-chicago/ | head
```

The second command should list top-level prefixes in the bucket.

## Diagnosing `SignatureDoesNotMatch`

If you see:

```
An error occurred (SignatureDoesNotMatch) when calling the ListObjectsV2 operation: ...
```

or

```
Unable to locate credentials
```

it means `~/.aws/credentials` either has no `[oci.chi]` section, or has placeholder/stale values. Fix:

1. Confirm the section name is exactly `oci.chi` (must match `[profile oci.chi]` in config, minus the `profile ` prefix).
2. Confirm the access key is a Customer Secret Key, not an OCI API key fingerprint/private key pair — those are a different credential type and will not work with the S3-compatible API.
3. Re-copy both `aws_access_key_id` and `aws_secret_access_key` together — a mismatched pair (old key + new secret, or vice versa) produces this exact error.

## Diagnosing region/endpoint errors

If you see an error mentioning the wrong region, or the request going to `s3.amazonaws.com` / `s3.us-chicago-1.amazonaws.com` instead of the OCI endpoint, the `services = oci-s3-compat-chi` line (or the `[services oci-s3-compat-chi]` block itself) is missing from `~/.aws/config`. Both the `[profile oci.chi]` block's `services =` line and the `[services oci-s3-compat-chi]` block are required together — the profile alone does nothing without the matching services block.

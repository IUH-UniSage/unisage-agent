"""Generates a throwaway test CA + a leaf cert for `fake-provider.test`.

Used only by the integration harness (plan.md "SSRF policy" - the fake
provider's TLS mode, and the DNS-rebinding test which needs a real TLS
handshake to distinguish "connected to the right IP" from "connected to
something else"). Never used outside `tests/e2e/` - the CA private key this
writes has no business existing anywhere near a real deployment.

Run standalone: `python -m tests.e2e.tls.generate_certs <output_dir>`. Skips
regenerating if the output dir already has both cert files (idempotent, so a
compose restart doesn't rotate the cert other containers already trust)
unless `--force` is passed.
"""

from __future__ import annotations

import argparse
import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

CA_KEY_FILE = "ca.key.pem"
CA_CERT_FILE = "ca.cert.pem"
SERVER_KEY_FILE = "server.key.pem"
SERVER_CERT_FILE = "server.cert.pem"
LEAF_HOSTNAME = "fake-provider.test"


def _write_key(path: Path, key: rsa.RSAPrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def _write_cert(path: Path, cert: x509.Certificate) -> None:
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def generate(output_dir: Path, force: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ca_key_path = output_dir / CA_KEY_FILE
    ca_cert_path = output_dir / CA_CERT_FILE
    server_key_path = output_dir / SERVER_KEY_FILE
    server_cert_path = output_dir / SERVER_CERT_FILE

    if not force and ca_cert_path.exists() and server_cert_path.exists():
        return

    now = datetime.datetime.now(datetime.UTC)
    not_before = now - datetime.timedelta(days=1)
    not_after = now + datetime.timedelta(days=365)

    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "unisage-agent integration test CA")]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write_key(ca_key_path, ca_key)
    _write_cert(ca_cert_path, ca_cert)

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, LEAF_HOSTNAME)])
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(LEAF_HOSTNAME), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    _write_key(server_key_path, server_key)
    _write_cert(server_cert_path, server_cert)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate(args.output_dir, force=args.force)
    print(f"Wrote CA + {LEAF_HOSTNAME} server cert to {args.output_dir}")


if __name__ == "__main__":
    main()

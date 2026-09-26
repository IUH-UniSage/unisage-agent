"""Ad hoc CA/leaf cert pairs for the Host/SNI/cert negative-case tests
(`tests/e2e/test_ssrf_host_sni_cert.py`).

Deliberately separate from `generate_certs.py`, which the real fake-provider
container also uses and which is intentionally narrow (one fixed CA, one
fixed hostname, idempotent-on-disk). The tests here need certs that are
*wrong* on purpose (mismatched hostname, expired, signed by an untrusted CA)
and want them purely in memory, generated fresh per test - none of that
belongs in the container's own cert-generation path.
"""

from __future__ import annotations

import datetime
import ipaddress
from dataclasses import dataclass

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


@dataclass(frozen=True)
class GeneratedCa:
    key_pem: bytes
    cert_pem: bytes
    _key: rsa.RSAPrivateKey
    _cert: x509.Certificate


@dataclass(frozen=True)
class GeneratedLeaf:
    key_pem: bytes
    cert_pem: bytes


def make_ca(common_name: str = "test CA") -> GeneratedCa:
    now = datetime.datetime.now(datetime.UTC)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
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
        .sign(key, hashes.SHA256())
    )
    return GeneratedCa(
        key_pem=key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        cert_pem=cert.public_bytes(serialization.Encoding.PEM),
        _key=key,
        _cert=cert,
    )


def make_leaf(
    ca: GeneratedCa,
    hostname: str,
    *,
    not_before: datetime.datetime | None = None,
    not_after: datetime.datetime | None = None,
    include_loopback_san: bool = True,
) -> GeneratedLeaf:
    """A leaf cert for `hostname`, signed by `ca`.

    `not_before`/`not_after` let a test build an expired (or not-yet-valid)
    cert on purpose; default is "valid for a year starting yesterday", same
    as `generate_certs.py`.
    """

    now = datetime.datetime.now(datetime.UTC)
    not_before = not_before if not_before is not None else now - datetime.timedelta(days=1)
    not_after = not_after if not_after is not None else now + datetime.timedelta(days=365)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    san = [x509.DNSName(hostname)]
    if include_loopback_san:
        san.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(ca._cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(ca._key, hashes.SHA256())
    )
    return GeneratedLeaf(
        key_pem=key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        cert_pem=cert.public_bytes(serialization.Encoding.PEM),
    )

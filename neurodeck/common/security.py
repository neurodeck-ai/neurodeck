"""TLS certificate generation and security utilities for NeuroDeck."""

import ssl
import socket
import secrets
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from datetime import datetime, timedelta
from typing import Tuple, Optional
import ipaddress

from .logging import get_logger

logger = get_logger("security")


class CertificateManager:
    """Manages TLS certificates for NeuroDeck communication."""
    
    def __init__(self, cert_dir: str = "config/certs"):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        
        self.cert_file = self.cert_dir / "server.crt"
        self.key_file = self.cert_dir / "server.key"
        self.ca_file = self.cert_dir / "ca.crt"
    
    def generate_self_signed_cert(
        self,
        hostname: str = "localhost",
        validity_days: int = 365
    ) -> Tuple[Path, Path]:
        """
        Generate self-signed certificate for development.
        
        Returns:
            Tuple of (cert_file_path, key_file_path)
        """
        logger.info(f"Generating self-signed certificate for {hostname}")
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Development"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NeuroDeck"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv6Address("::1")),
            ]),
            critical=False,
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                key_encipherment=True,
                digital_signature=True,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                encipher_only=False,
                decipher_only=False
            ),
            critical=True,
        ).add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=True,
        ).sign(private_key, hashes.SHA256())
        
        # Write certificate
        with open(self.cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        # Write private key
        with open(self.key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Set appropriate permissions (owner read/write only)
        self.cert_file.chmod(0o600)
        self.key_file.chmod(0o600)
        
        logger.info(f"Certificate generated: {self.cert_file}")
        logger.info(f"Private key generated: {self.key_file}")
        
        return self.cert_file, self.key_file
    
    def create_ssl_context(
        self,
        is_server: bool = True,
        verify_mode: ssl.VerifyMode = ssl.CERT_NONE
    ) -> ssl.SSLContext:
        """
        Create SSL context for TLS connections.
        
        Args:
            is_server: True for server context, False for client
            verify_mode: Certificate verification mode
        """
        if is_server:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.cert_file, self.key_file)
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False  # For self-signed certs
        
        context.verify_mode = verify_mode
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Security settings
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        return context
    
    def ensure_certificates_exist(self, hostname: str = "localhost") -> bool:
        """
        Ensure certificates exist, generate if missing.
        
        Returns:
            True if certificates are ready
        """
        if not self.cert_file.exists() or not self.key_file.exists():
            logger.info("Certificates not found, generating new self-signed certificates")
            self.generate_self_signed_cert(hostname)
            return True
        
        # Check if certificates are still valid
        try:
            with open(self.cert_file, 'rb') as f:
                cert = x509.load_pem_x509_certificate(f.read())
            
            if cert.not_valid_after < datetime.utcnow():
                logger.warning("Certificate expired, generating new one")
                self.generate_self_signed_cert(hostname)
                return True
            
            logger.info("Using existing valid certificates")
            return True
            
        except Exception as e:
            logger.error(f"Error reading certificate: {e}")
            logger.info("Regenerating certificates")
            self.generate_self_signed_cert(hostname)
            return True


class TokenManager:
    """Manages authentication tokens for NeuroDeck."""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def validate_token(provided_token: str, expected_token: str) -> bool:
        """
        Validate authentication token using constant-time comparison.
        
        Args:
            provided_token: Token provided by client
            expected_token: Expected token from configuration
            
        Returns:
            True if tokens match
        """
        if not provided_token or not expected_token:
            return False
        
        return secrets.compare_digest(provided_token, expected_token)


def test_tls_connection(host: str = "localhost", port: int = 9999) -> bool:
    """
    Test TLS connection to verify certificate setup.
    
    Returns:
        True if connection successful
    """
    try:
        cert_manager = CertificateManager()
        context = cert_manager.create_ssl_context(is_server=False)
        
        with socket.create_connection((host, port), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                logger.info(f"TLS connection successful to {host}:{port}")
                logger.info(f"Cipher: {ssock.cipher()}")
                logger.info(f"Protocol: {ssock.version()}")
                return True
                
    except Exception as e:
        logger.error(f"TLS connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test certificate generation
    cert_manager = CertificateManager("../config/certs")
    cert_manager.ensure_certificates_exist()
    
    # Test token generation
    token = TokenManager.generate_secure_token()
    print(f"Generated token: {token}")
    
    # Test token validation
    is_valid = TokenManager.validate_token(token, token)
    print(f"Token validation: {is_valid}")
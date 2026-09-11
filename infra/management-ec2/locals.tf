locals {
  name_prefix                = "capitonic-polymarket-bot-management"
  tunnel_token_parameter     = "/capitonic/production/management/cloudflare-tunnel-token"
  ubuntu_ami_parameter       = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
  cloudflare_tunnel_dns_name = "${cloudflare_zero_trust_tunnel_cloudflared.management.id}.cfargotunnel.com"

  common_tags = {
    Application = "capitonic"
    Component   = "management"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

resource "cloudflare_zero_trust_tunnel_cloudflared" "management" {
  account_id = var.cloudflare_account_id
  name       = "${local.name_prefix}-rdp"
  config_src = "cloudflare"
}

data "cloudflare_zero_trust_tunnel_cloudflared_token" "management" {
  account_id = var.cloudflare_account_id
  tunnel_id  = cloudflare_zero_trust_tunnel_cloudflared.management.id
}

resource "cloudflare_zero_trust_tunnel_cloudflared_config" "management" {
  account_id = var.cloudflare_account_id
  tunnel_id  = cloudflare_zero_trust_tunnel_cloudflared.management.id
  source     = "cloudflare"

  config = {
    ingress = [
      {
        hostname = var.rdp_hostname
        service  = "rdp://127.0.0.1:3389"
      },
      {
        service = "http_status:404"
      }
    ]
  }
}

resource "cloudflare_dns_record" "management_rdp" {
  zone_id = var.cloudflare_zone_id
  name    = var.rdp_hostname
  content = local.cloudflare_tunnel_dns_name
  type    = "CNAME"
  ttl     = 1
  proxied = true
  comment = "Terraform-managed private RDP entrypoint for the Capitonic production management host"
}

resource "cloudflare_zero_trust_access_application" "management_rdp" {
  zone_id          = var.cloudflare_zone_id
  name             = "Capitonic production management RDP"
  domain           = var.rdp_hostname
  type             = "self_hosted"
  session_duration = "8h"

  policies = [{
    name       = "Allow management operator"
    decision   = "allow"
    precedence = 1
    include = [{
      email = {
        email = var.cloudflare_access_email
      }
    }]
  }]
}

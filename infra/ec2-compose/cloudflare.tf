provider "cloudflare" {}

locals {
  cloudflare_tunnel_dns_target = "${var.cloudflare_tunnel_id}.cfargotunnel.com"
}

resource "cloudflare_dns_record" "ssh_ops" {
  zone_id = var.cloudflare_zone_id
  name    = var.cloudflare_ssh_hostname
  content = local.cloudflare_tunnel_dns_target
  type    = "CNAME"
  ttl     = 1
  proxied = true
  comment = "Managed by Terraform for ${var.name} ${var.environment} SSH tunnel access"
}

resource "cloudflare_dns_record" "rdp_ops" {
  zone_id = var.cloudflare_zone_id
  name    = var.cloudflare_rdp_hostname
  content = local.cloudflare_tunnel_dns_target
  type    = "CNAME"
  ttl     = 1
  proxied = true
  comment = "Managed by Terraform for ${var.name} ${var.environment} RDP tunnel access"
}

output "instance_id" {
  description = "Management EC2 instance ID."
  value       = aws_instance.management.id
}

output "rdp_hostname" {
  description = "Cloudflare Access RDP hostname."
  value       = var.rdp_hostname
}

output "tunnel_id" {
  description = "Dedicated Cloudflare Tunnel ID."
  value       = cloudflare_zero_trust_tunnel_cloudflared.management.id
}

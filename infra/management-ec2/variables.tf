variable "aws_region" {
  description = "AWS region for every management-host resource."
  type        = string
  default     = "eu-west-1"

  validation {
    condition     = var.aws_region == "eu-west-1"
    error_message = "The production management host must be deployed in eu-west-1 (Ireland)."
  }
}

variable "instance_type" {
  description = "Burstable management host size suitable for an idle XFCE/Chrome desktop."
  type        = string
  default     = "t3.medium"
}

variable "root_volume_size_gib" {
  description = "Encrypted gp3 root volume size."
  type        = number
  default     = 24
}

variable "app_secret_name" {
  description = "Unified production secret containing RDP_PASSWORD."
  type        = string
  default     = "capitonic/polymarket-bot/production"
}

variable "rdp_username" {
  description = "Non-root XRDP login user."
  type        = string
  default     = "capitonic"
}

variable "cloudflare_account_id" {
  description = "Cloudflare account containing the tunnel and Access application."
  type        = string
  sensitive   = true
}

variable "cloudflare_zone_id" {
  description = "Cloudflare zone ID for capitonic.com."
  type        = string
  sensitive   = true
}

variable "cloudflare_access_email" {
  description = "Only identity allowed by the management RDP Access policy."
  type        = string
  sensitive   = true
}

variable "rdp_hostname" {
  description = "Cloudflare Access hostname for management RDP."
  type        = string
  default     = "rdp.capitonic.com"
}

variable "aws_region" {
  description = "AWS region where the EC2 Docker Compose host is created."
  type        = string
  default     = "mx-central-1"
}

variable "name" {
  description = "Name prefix for stack resources."
  type        = string
  default     = "capitonic-polymarket-bot"
}

variable "environment" {
  description = "Environment tag value."
  type        = string
  default     = "production"
}

variable "instance_type" {
  description = "EC2 instance type. t3.medium is 2 vCPU and 4 GiB RAM."
  type        = string
  default     = "t3.medium"
}

variable "ami_id" {
  description = "Optional AMI override. Leave null to use latest Ubuntu 24.04 LTS x86_64."
  type        = string
  default     = null
}

variable "subnet_id" {
  description = "Optional subnet override. Leave null to use the first default subnet in the default VPC."
  type        = string
  default     = null
}

variable "ssh_key_name" {
  description = "Optional existing EC2 key pair name for SSH access."
  type        = string
  default     = null
}

variable "root_volume_size_gib" {
  description = "Root EBS volume size in GiB."
  type        = number
  default     = 50

  validation {
    condition     = var.root_volume_size_gib >= 20
    error_message = "root_volume_size_gib must be at least 20 GiB."
  }
}

variable "repo_url" {
  description = "GitHub HTTPS repository URL cloned by the EC2 bootstrap."
  type        = string
  default     = "https://github.com/gunnermadsen/polymarket-trading-bot.git"
}

variable "repo_branch" {
  description = "Git branch cloned by the EC2 bootstrap."
  type        = string
  default     = "production"
}

variable "app_directory" {
  description = "Directory where the repository is cloned on the EC2 host."
  type        = string
  default     = "/opt/polymarket-bot"
}

variable "compose_file" {
  description = "Docker Compose file used by the EC2 bootstrap."
  type        = string
  default     = "docker-compose.production.yml"
}

variable "app_secret_name" {
  description = "AWS Secrets Manager JSON secret containing app secrets and GitHub clone credentials."
  type        = string
  default     = "capitonic/polymarket-bot/production"
}

variable "ecr_registry" {
  description = "ECR registry host used by Docker Compose image references."
  type        = string
  default     = "192200846560.dkr.ecr.mx-central-1.amazonaws.com"
}

variable "cloudflare_zone_id" {
  description = "Cloudflare zone ID for capitonic.com."
  type        = string
  sensitive   = true
}

variable "cloudflare_tunnel_id" {
  description = "Cloudflare Tunnel UUID used for SSH/RDP CNAME targets."
  type        = string
  sensitive   = true
}

variable "cloudflare_ssh_hostname" {
  description = "Cloudflare Access SSH hostname routed to the EC2 cloudflared daemon."
  type        = string
  default     = "ssh.ops.capitonic.com"
}

variable "cloudflare_rdp_hostname" {
  description = "Cloudflare Access RDP hostname routed to the EC2 cloudflared daemon."
  type        = string
  default     = "rdp.ops.capitonic.com"
}

variable "enable_desktop" {
  description = "Install XFCE, XRDP, and a lightweight browser on the host."
  type        = bool
  default     = true
}

variable "rdp_username" {
  description = "Local Linux username for XRDP logins. The password is read from RDP_PASSWORD in AWS Secrets Manager."
  type        = string
  default     = "polybot"
}

variable "enable_cloudflared" {
  description = "Install and run cloudflared as a systemd service. Tunnel credentials are read from AWS Secrets Manager."
  type        = bool
  default     = true
}

variable "cloudflare_tunnel_ipv4_cidrs" {
  description = "Cloudflare Tunnel edge IPv4 CIDRs allowed for outbound TCP/7844."
  type        = list(string)
  default = [
    "198.41.192.7/32",
    "198.41.192.27/32",
    "198.41.192.37/32",
    "198.41.192.47/32",
    "198.41.192.57/32",
    "198.41.192.67/32",
    "198.41.192.77/32",
    "198.41.192.107/32",
    "198.41.192.167/32",
    "198.41.192.227/32",
    "198.41.200.13/32",
    "198.41.200.23/32",
    "198.41.200.33/32",
    "198.41.200.43/32",
    "198.41.200.53/32",
    "198.41.200.63/32",
    "198.41.200.73/32",
    "198.41.200.113/32",
    "198.41.200.193/32",
    "198.41.200.233/32",
  ]
}

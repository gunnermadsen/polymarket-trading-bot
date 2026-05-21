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
  description = "Optional AMI override. Leave null to use latest Amazon Linux 2023 x86_64."
  type        = string
  default     = null
}

variable "subnet_id" {
  description = "Optional subnet override. Leave null to use the first default subnet in the default VPC."
  type        = string
  default     = null
}

variable "admin_cidrs" {
  description = "Optional CIDR blocks allowed to SSH to the host."
  type        = list(string)
  default     = []
}

variable "ssh_key_name" {
  description = "Optional existing EC2 key pair name for SSH access."
  type        = string
  default     = null
}

variable "root_volume_size_gib" {
  description = "Root EBS volume size in GiB."
  type        = number
  default     = 30

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

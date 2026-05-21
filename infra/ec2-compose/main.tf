provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  filter {
    name   = "isDefault"
    values = ["true"]
  }
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

data "aws_ssm_parameter" "ubuntu_ami" {
  name = local.ami_ssm_parameter_name
}

data "aws_secretsmanager_secret" "app" {
  name = var.app_secret_name
}

resource "aws_iam_role" "compose_host" {
  name = "${local.name_prefix}-host"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "compose_host" {
  name = "${local.name_prefix}-host"
  role = aws_iam_role.compose_host.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = data.aws_secretsmanager_secret.app.arn
      },
      {
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:DescribeImages",
          "ecr:DescribeRepositories",
          "ecr:GetDownloadUrlForLayer"
        ]
        Resource = [
          "arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/capitonic/polymarket-bot",
          "arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/capitonic/db-migrate"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "compose_host_ssm" {
  role       = aws_iam_role.compose_host.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "compose_host" {
  name = "${local.name_prefix}-host"
  role = aws_iam_role.compose_host.name
}

resource "aws_security_group" "compose_host" {
  name        = "${local.name_prefix}-sg"
  description = "Access for the Polymarket Docker Compose host"
  vpc_id      = data.aws_vpc.default.id

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-sg"
  })
}

resource "aws_vpc_security_group_egress_rule" "https_ipv4" {
  security_group_id = aws_security_group.compose_host.id
  description       = "HTTPS egress for AWS APIs, GitHub, ECR, Docker, package repositories, and Cloudflare Tunnel"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  ip_protocol       = "tcp"
  to_port           = 443
}

resource "aws_vpc_security_group_egress_rule" "cloudflare_tunnel_http2_ipv4" {
  for_each = toset(var.cloudflare_tunnel_ipv4_cidrs)

  security_group_id = aws_security_group.compose_host.id
  description       = "Cloudflare Tunnel http2 egress to ${each.value}"
  cidr_ipv4         = each.value
  from_port         = 7844
  ip_protocol       = "tcp"
  to_port           = 7844
}

resource "aws_vpc_security_group_egress_rule" "dns_udp_ipv4" {
  security_group_id = aws_security_group.compose_host.id
  description       = "DNS egress to VPC resolver"
  cidr_ipv4         = data.aws_vpc.default.cidr_block
  from_port         = 53
  ip_protocol       = "udp"
  to_port           = 53
}

resource "aws_vpc_security_group_egress_rule" "dns_tcp_ipv4" {
  security_group_id = aws_security_group.compose_host.id
  description       = "DNS TCP egress to VPC resolver"
  cidr_ipv4         = data.aws_vpc.default.cidr_block
  from_port         = 53
  ip_protocol       = "tcp"
  to_port           = 53
}

resource "aws_instance" "compose_host" {
  ami                         = local.ami_id
  instance_type               = var.instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.compose_host.id]
  key_name                    = var.ssh_key_name
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.compose_host.name
  user_data_replace_on_change = true

  user_data = templatefile("${path.module}/templates/user-data.sh.tftpl", {
    aws_region         = var.aws_region
    app_directory      = var.app_directory
    app_secret_name    = var.app_secret_name
    ecr_registry       = var.ecr_registry
    enable_cloudflared = var.enable_cloudflared
    enable_desktop     = var.enable_desktop
    rdp_username       = var.rdp_username
    repo_branch        = var.repo_branch
    repo_url           = var.repo_url
  })

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size_gib
    encrypted             = true
    delete_on_termination = true
  }

  tags = merge(local.common_tags, {
    Name = local.name_prefix
  })

  lifecycle {
    precondition {
      condition     = local.subnet_id != null
      error_message = "No default subnet was found. Set subnet_id explicitly or create a default subnet in the default VPC."
    }
  }
}

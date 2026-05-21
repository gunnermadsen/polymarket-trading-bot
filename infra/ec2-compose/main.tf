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

data "aws_ssm_parameter" "al2023_ami" {
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

resource "aws_vpc_security_group_ingress_rule" "admin" {
  for_each = local.admin_ingress_rules

  security_group_id = aws_security_group.compose_host.id
  description       = each.value.description
  cidr_ipv4         = each.value.cidr
  from_port         = each.value.from_port
  ip_protocol       = "tcp"
  to_port           = each.value.to_port
}

resource "aws_vpc_security_group_egress_rule" "all_ipv4" {
  security_group_id = aws_security_group.compose_host.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
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
    aws_region      = var.aws_region
    app_directory   = var.app_directory
    app_secret_name = var.app_secret_name
    ecr_registry    = var.ecr_registry
    repo_branch     = var.repo_branch
    repo_url        = var.repo_url
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

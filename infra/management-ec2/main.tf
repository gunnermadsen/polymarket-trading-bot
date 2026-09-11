data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
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
  name = local.ubuntu_ami_parameter
}

data "aws_secretsmanager_secret" "production" {
  name = var.app_secret_name
}

resource "aws_ssm_parameter" "tunnel_token" {
  name        = local.tunnel_token_parameter
  description = "Ephemeral Cloudflare connector token for the production management host"
  type        = "SecureString"
  value       = data.cloudflare_zero_trust_tunnel_cloudflared_token.management.token

  tags = local.common_tags
}

resource "aws_iam_role" "management" {
  name = "${local.name_prefix}-host"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "management" {
  name = "${local.name_prefix}-bootstrap"
  role = aws_iam_role.management.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:DescribeSecret",
          "secretsmanager:GetSecretValue"
        ]
        Resource = data.aws_secretsmanager_secret.production.arn
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter"
        ]
        Resource = aws_ssm_parameter.tunnel_token.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.management.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "management" {
  name = "${local.name_prefix}-host"
  role = aws_iam_role.management.name
}

resource "aws_security_group" "management" {
  name        = "${local.name_prefix}-sg"
  description = "Outbound-only access for the Capitonic management host"
  vpc_id      = data.aws_vpc.default.id

  tags = merge(local.common_tags, { Name = "${local.name_prefix}-sg" })
}

resource "aws_vpc_security_group_egress_rule" "https" {
  security_group_id = aws_security_group.management.id
  description       = "HTTPS for AWS APIs, package repositories, Chrome, and Cloudflare Tunnel"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 443
  ip_protocol       = "tcp"
  to_port           = 443
}

resource "aws_vpc_security_group_egress_rule" "cloudflare_quic" {
  security_group_id = aws_security_group.management.id
  description       = "Cloudflare Tunnel QUIC"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 7844
  ip_protocol       = "udp"
  to_port           = 7844
}

resource "aws_vpc_security_group_egress_rule" "cloudflare_http2" {
  security_group_id = aws_security_group.management.id
  description       = "Cloudflare Tunnel HTTP/2 fallback"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 7844
  ip_protocol       = "tcp"
  to_port           = 7844
}

resource "aws_vpc_security_group_egress_rule" "dns_udp" {
  security_group_id = aws_security_group.management.id
  description       = "DNS through the VPC resolver"
  cidr_ipv4         = data.aws_vpc.default.cidr_block
  from_port         = 53
  ip_protocol       = "udp"
  to_port           = 53
}

resource "aws_vpc_security_group_egress_rule" "dns_tcp" {
  security_group_id = aws_security_group.management.id
  description       = "DNS TCP through the VPC resolver"
  cidr_ipv4         = data.aws_vpc.default.cidr_block
  from_port         = 53
  ip_protocol       = "tcp"
  to_port           = 53
}

resource "aws_vpc_security_group_egress_rule" "ntp" {
  security_group_id = aws_security_group.management.id
  description       = "NTP for reliable authentication timestamps"
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 123
  ip_protocol       = "udp"
  to_port           = 123
}

resource "aws_instance" "management" {
  ami                         = data.aws_ssm_parameter.ubuntu_ami.value
  instance_type               = var.instance_type
  subnet_id                   = sort(data.aws_subnets.default.ids)[0]
  vpc_security_group_ids      = [aws_security_group.management.id]
  associate_public_ip_address = true
  iam_instance_profile        = aws_iam_instance_profile.management.name
  user_data_replace_on_change = true

  user_data = templatefile("${path.module}/templates/user-data.sh.tftpl", {
    app_secret_name        = var.app_secret_name
    aws_region             = var.aws_region
    rdp_username           = var.rdp_username
    tunnel_token_parameter = aws_ssm_parameter.tunnel_token.name
  })

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = var.root_volume_size_gib
    encrypted             = true
    delete_on_termination = true
  }

  tags = merge(local.common_tags, { Name = "${local.name_prefix}-host" })

  depends_on = [
    cloudflare_zero_trust_tunnel_cloudflared_config.management,
    aws_iam_role_policy.management,
    aws_iam_role_policy_attachment.ssm_core
  ]
}

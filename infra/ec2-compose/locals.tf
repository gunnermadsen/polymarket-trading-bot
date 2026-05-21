locals {
  common_tags = {
    Project     = var.name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }

  name_prefix            = var.name
  ami_ssm_parameter_name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
  ami_id                 = coalesce(var.ami_id, data.aws_ssm_parameter.al2023_ami.value)
  subnet_id              = coalesce(var.subnet_id, try(sort(data.aws_subnets.default.ids)[0], null))

  admin_ingress_rules = {
    for rule in flatten([
      for cidr in var.admin_cidrs : [
        {
          key         = "ssh-${replace(cidr, "/", "-")}"
          description = "SSH from ${cidr}"
          cidr        = cidr
          from_port   = 22
          to_port     = 22
        }
      ]
    ]) : rule.key => rule
  }
}

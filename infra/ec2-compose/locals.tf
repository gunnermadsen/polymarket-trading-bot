locals {
  common_tags = {
    Project     = var.name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }

  name_prefix            = var.name
  ami_ssm_parameter_name = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
  ami_id                 = coalesce(var.ami_id, data.aws_ssm_parameter.ubuntu_ami.value)
  subnet_id              = coalesce(var.subnet_id, try(sort(data.aws_subnets.default.ids)[0], null))
}

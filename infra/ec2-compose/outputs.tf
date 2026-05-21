output "instance_id" {
  description = "EC2 instance ID for the Docker Compose host."
  value       = aws_instance.compose_host.id
}

output "public_ip" {
  description = "Public IPv4 address of the Docker Compose host."
  value       = aws_instance.compose_host.public_ip
}

output "public_dns" {
  description = "Public DNS name of the Docker Compose host."
  value       = aws_instance.compose_host.public_dns
}

output "ssm_start_session_command" {
  description = "Command to open an SSM shell on the host."
  value       = "aws ssm start-session --region ${var.aws_region} --target ${aws_instance.compose_host.id}"
}

output "app_status_command" {
  description = "Command to inspect the Compose app over SSM."
  value       = "aws ssm start-session --region ${var.aws_region} --target ${aws_instance.compose_host.id}"
}

output "app_secret_name" {
  description = "Secrets Manager secret consumed by the host bootstrap."
  value       = var.app_secret_name
}

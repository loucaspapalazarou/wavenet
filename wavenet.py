import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            bias=False,
        )

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, : -self.padding]
        # return x[:, :, : -self.padding]


class CausalConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations):
        super(CausalConv1dBlock, self).__init__()
        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(
                CausalConv1d(in_channels, out_channels, kernel_size, dilation)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stack_size) -> None:
        super(ResidualBlock, self).__init__()
        dilations = [2**s for s in range(stack_size)]
        self.causal_block = CausalConv1dBlock(
            in_channels, out_channels, kernel_size, dilations
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_init = self.residual_conv(x)
        x = self.causal_block(x)
        x_gated = self.tanh(x) * self.sigmoid(x)
        x_conv = self.conv(x_gated)

        skip_output = x_conv
        block_output = x_conv + x_init

        return block_output, skip_output


class DenseNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DenseNet, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return self.softmax(x)


class WaveNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stack_size, num_layers
    ) -> None:
        super(WaveNet, self).__init__()
        self.causal_block = CausalConv1dBlock(
            in_channels, out_channels, kernel_size, dilations=[1] * stack_size
        )
        self.residual_block_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.residual_block_layers.append(
                ResidualBlock(out_channels, out_channels, kernel_size, stack_size)
            )
        self.dense_net = DenseNet(
            out_channels,
            out_channels,
        )

    def forward(self, x):
        output = self.causal_block(x)
        s = torch.zeros_like(output)
        for res_block in self.residual_block_layers:
            output, skip = res_block(output)
            s += skip
        return self.dense_net(s)

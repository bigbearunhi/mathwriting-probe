import unittest
import torch
from probe import features, Model


class ProbeChecks(unittest.TestCase):
    def test_stroke_boundaries_and_aspect_ratio(self):
        sample = {'strokes': [[[0, 0, 0], [2, 0, 1], [4, 0, 2]],
                              [[4, 2, 3], [4, 4, 4]] ]}
        x = features(sample, max_points=4)
        self.assertEqual(x.shape, (4, 3))
        self.assertEqual(x[:, 2].tolist(), [0, 1, 0, 1])
        self.assertTrue(torch.allclose(x[0, :2], torch.tensor([-0.5, -0.5])))
        self.assertTrue(torch.allclose(x[-1, :2], torch.tensor([0.5, 0.5])))

    def test_no_future_or_padding_leakage(self):
        torch.manual_seed(3)
        model = Model(12, dim=32, layers=1, heads=4).eval()
        src = torch.randn(1, 5, 3)
        mask = torch.tensor([[False, False, False, True, True]])
        target = torch.tensor([[1, 4, 5, 6]])
        with torch.no_grad():
            first = model(src, target, mask)
            target[0, 2:] = torch.tensor([8, 9])
            src[0, 3:] = 100
            second = model(src, target, mask)
        self.assertTrue(torch.allclose(first[:, :2], second[:, :2], atol=1e-5))


if __name__ == '__main__':
    unittest.main()

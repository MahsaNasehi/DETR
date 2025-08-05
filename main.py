import config
from dataset.build import get_dataset, collate_fn
from torch.utils.data import DataLoader
from engine.trainer import train
from torchvision import transforms
from models.detr import get_model
from engine.optimizer import get_optimizer

def main():
    transform = config.get_transform()
    args = config.get_args()
    batch_size = args['batch_size']
    data_path = args['data_path']
    annotation_file = args['annotation_file']
    device = args['device']
    dataset = get_dataset(data_path, annotation_file, transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn)
    model = get_model(num_classes=args['num_classes'], num_queries=args['num_query'], d_model=args['d_model'], device=device)
    optimizer = get_optimizer(model, name='adamw', lr=args['lr'], weight_decay=1e-4)
    train(args['epochs'], model, data_loader, optimizer, device)

if __name__ == "__main__":
    main()
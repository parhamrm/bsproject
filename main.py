from utils.options import parser
from utils.loader import load_model, load_image, tensor2im
from PIL import Image

if __name__ == "__main__":
    args = parser.parse_args()
    model = load_model(args.model)
    model.setup()

    image = load_image(args.image, args.model)

    model.set_input(image)
    model.test()

    fake = model.get_generated()
    fake = tensor2im(fake)

    image_pil = Image.fromarray(fake)
    image_pil.save(f'{args.save_to}/{args.model}.jpg')
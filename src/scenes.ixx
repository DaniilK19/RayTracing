// Module scenes: scenes, i.e. what to render (objects) and how to shoot it (camera settings).
export module scenes;

import std;
import camera;
import hittable;
import material;
import random;
import sphere;
import vec3;

// A scene: objects and the camera that shoots them.
export struct scene
{
    hittable_list world;
    camera_settings camera;
};

// A scene in the list: a name for the command line, a description for the help and the function that builds it.
// A function pointer is stored rather than a ready scene: only the chosen scene gets built.
export struct scene_entry
{
    std::string_view name;
    std::string_view description;
    scene (*build)();
};

// The scene from the first book's cover: three big spheres and a scatter of small ones with random materials.
// The main thread's random number generator always starts from the same seed, so the scene
// is the same on every run.
scene weekend()
{
    hittable_list world;

    const auto ground_material = std::make_shared<lambertian>(color{0.5, 0.5, 0.5});
    world.add(std::make_unique<sphere>(point3{0, -1000, 0}, 1000, ground_material));

    for (int a = -11; a < 11; ++a)
    {
        for (int b = -11; b < 11; ++b)
        {
            const double choose_mat = random_double();
            const point3 center{a + 0.9 * random_double(), 0.2, b + 0.9 * random_double()};
            if ((center - point3{4, 0.2, 0}).length() <= 0.9)
            {
                continue; // don't put a small sphere right next to the big metal one
            }

            std::shared_ptr<material> sphere_material;
            if (choose_mat < 0.8) // 80% matte
            {
                sphere_material = std::make_shared<lambertian>(color::random() * color::random());
            }
            else if (choose_mat < 0.95) // 15% metal
            {
                sphere_material = std::make_shared<metal>(color::random(0.5, 1), random_double(0, 0.5));
            }
            else // 5% glass
            {
                sphere_material = std::make_shared<dielectric>(1.5);
            }
            world.add(std::make_unique<sphere>(center, 0.2, std::move(sphere_material)));
        }
    }

    world.add(std::make_unique<sphere>(point3{0, 1, 0}, 1.0, std::make_shared<dielectric>(1.5)));
    world.add(std::make_unique<sphere>(point3{-4, 1, 0}, 1.0, std::make_shared<lambertian>(color{0.4, 0.2, 0.1})));
    world.add(std::make_unique<sphere>(point3{4, 1, 0}, 1.0, std::make_shared<metal>(color{0.7, 0.6, 0.5}, 0.0)));

    return {std::move(world),
            {
                .aspect_ratio = 16.0 / 9.0,
                .image_width = 1920, // Full HD: with a 16:9 aspect ratio the height will be 1080
                .samples_per_pixel = 500,
                .max_depth = 50,
                .vfov = 20,
                .lookfrom = {13, 2, 3},
                .lookat = {0, 0, 0},
                .vup = {0, 1, 0},
                .defocus_angle = 0.6,
                .focus_dist = 10.0,
            }};
}

// All scenes, in the order they appear in the books.
export constexpr std::array scene_list{
    scene_entry{"weekend", "the first book's cover: spheres of three materials", weekend},
};

export constexpr std::string_view default_scene = "weekend";

// The scene with the given name, or nullptr if there is none. The projection &scene_entry::name (C++20):
// std::ranges::find compares name not with the elements themselves but with their name field.
export const scene_entry* find_scene(std::string_view name)
{
    const auto it = std::ranges::find(scene_list, name, &scene_entry::name);
    return it != scene_list.end() ? &*it : nullptr;
}

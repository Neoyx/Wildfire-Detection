#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <set>
#include <map>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace py = pybind11;

struct UnionFind
{
    std::vector<int> parent;

    UnionFind(int n)
    {
        parent.resize(n);
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int i)
    {
        if (parent[i] == i)
            return i;
        return parent[i] = find(parent[i]); // Path compression
    }

    void unite(int i, int j)
    {
        int root_i = find(i);
        int root_j = find(j);
        if (root_i != root_j)
        {
            parent[root_i] = root_j;
        }
    }
};

std::tuple<py::array_t<uint16_t>, size_t> sequential_regioning_cpp(py::array_t<uint16_t> img_py_in, bool n8, int random_seed)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    // Request buffer info from the input array
    py::buffer_info buf_in = img_py_in.request();
    if (buf_in.ndim != 2)
    {
        throw std::runtime_error("Input image must be 2-dimensional.");
    }

    py::ssize_t height = buf_in.shape[0];
    py::ssize_t width = buf_in.shape[1];

    // Create a mutable copy of the input image for labeling
    // This explicitly creates a new array and copies the data, ensuring it's writable.
    py::array_t<uint16_t> img_labels({height, width});
    std::memcpy(static_cast<uint16_t *>(img_labels.request().ptr),
                static_cast<const uint16_t *>(buf_in.ptr),
                height * width * sizeof(uint16_t));

    int next_label = 2;
    std::vector<std::pair<int, int>> equivalences;

    for (py::ssize_t v = 0; v < height; ++v)
    {
        for (py::ssize_t u = 0; u < width; ++u)
        {
            if (img_labels.at(v, u) == 1)
            {
                std::set<int> neighbors_labels;

                if (u > 0 && img_labels.at(v, u - 1) > 1) // Left
                    neighbors_labels.insert(img_labels.at(v, u - 1));
                if (v > 0 && img_labels.at(v - 1, u) > 1) // Top
                    neighbors_labels.insert(img_labels.at(v - 1, u));

                if (n8)
                {
                    if (v > 0 && u > 0 && img_labels.at(v - 1, u - 1) > 1) // Top-Left
                        neighbors_labels.insert(img_labels.at(v - 1, u - 1));
                    if (v > 0 && u < width - 1 && img_labels.at(v - 1, u + 1) > 1) // Top-Right
                        neighbors_labels.insert(img_labels.at(v - 1, u + 1));
                }

                if (neighbors_labels.empty())
                {
                    img_labels.mutable_at(v, u) = next_label++;
                }
                else
                {
                    int min_label = *neighbors_labels.begin();
                    img_labels.mutable_at(v, u) = min_label;

                    // Record equivalences if multiple neighbor labels are found
                    for (int label : neighbors_labels)
                    {
                        if (label != min_label)
                        {
                            equivalences.push_back({min_label, label});
                        }
                    }
                }
            }
        }
    }

    UnionFind uf(next_label); // next_label is the size needed for UF (max label + 1)
    for (const auto &col : equivalences)
    {
        uf.unite(col.first, col.second);
    }

    // Collect unique representative labels and assign colors
    std::map<int, std::vector<uint16_t>> representative_to_color;
    std::mt19937 gen(random_seed);
    std::uniform_int_distribution<> distrib(50, 255);

    py::array_t<uint16_t> out_img_py({height, width, (py::ssize_t)3});

    std::set<int> unique_root_labels;

    // Second Pass: Relabeling and Coloring
    for (py::ssize_t v = 0; v < height; ++v)
    {
        for (py::ssize_t u = 0; u < width; ++u)
        {
            uint16_t current_label = img_labels.at(v, u);

            if (current_label > 1)
            {
                int representative_label = uf.find(current_label);
                unique_root_labels.insert(representative_label); // Track unique roots for final count

                if (representative_to_color.find(representative_label) == representative_to_color.end())
                {
                    representative_to_color[representative_label] = {
                        static_cast<uint16_t>(distrib(gen)),
                        static_cast<uint16_t>(distrib(gen)),
                        static_cast<uint16_t>(distrib(gen))};
                }

                out_img_py.mutable_at(v, u, 0) = representative_to_color[representative_label][0];
                out_img_py.mutable_at(v, u, 1) = representative_to_color[representative_label][1];
                out_img_py.mutable_at(v, u, 2) = representative_to_color[representative_label][2];
            }
        }
    }
    size_t num_regions = unique_root_labels.size();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "sequential regioning:     Regions found: " << num_regions << std::endl;
    std::cout << "sequential regioning:     Elapsed time: " << elapsed_ms.count() << " milliseconds" << std::endl;
    return std::make_tuple(out_img_py, num_regions);
}

PYBIND11_MODULE(sequential_regioning_cpp, m)
{
    m.doc() = "A fast C++ implementation of sequential regioning, callable from Python.";
    m.def("run", &sequential_regioning_cpp, "Performs sequential regioning on a 2D numpy array.",
          py::arg("img"), py::arg("n8"), py::arg("random_seed") = 20);
}
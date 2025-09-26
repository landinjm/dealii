// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2012 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// this tests whether Portable::FEEvaluation::read_dof_values and
// Portable::FEEvaluation::read_dof_values_plain get the same data
// on a vector where constraints are distributed

#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

template <int dim, int fe_degree, typename number>
void
do_test(const DoFHandler<dim>           &dof,
        const AffineConstraints<number> &constraints)
{
  deallog << "Testing " << dof.get_fe().get_name() << std::endl;

  Vector<number> solution(dof.n_dofs());

  // fill vector with random entries
  for (unsigned int i = 0; i < dof.n_dofs(); ++i)
    {
      if (constraints.is_constrained(i))
        continue;
      const double entry = random_value<double>();
      solution(i)        = entry;
    }

  constraints.distribute(solution);
  Portable::MatrixFree<dim, number> mf_data;
  {
    const QGauss<1> quad(fe_degree + 1);
    typename Portable::MatrixFree<dim, number>::AdditionalData data;
  }
}

template <int dim, int fe_degree>
void
test()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_shell(tria, Point<dim>(), 1., 2., 96, true);

  // refine a few cells
  for (unsigned int i = 0; i < 11 - 3 * dim; ++i)
    {
      unsigned int counter = 0;
      for (const auto &cell : tria.active_cell_iterators())
        {
          if (counter % (7 - i) == 0)
            cell->set_refine_flag();
          ++counter;
        }
      tria.execute_coarsening_and_refinement();
    }

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof, constraints);
  VectorTools::interpolate_boundary_values(dof,
                                           1,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  do_test<dim, fe_degree, double>(dof, constraints);
}

int
main()
{
  std::ofstream logfile("output");
  deallog.attach(logfile);

  Kokkos::initialize();

  deallog << std::fixed << std::setprecision(3);

  {
    deallog.push("2d");
    test<2, 2>();
    deallog.pop();
    deallog.push("3d");
    test<3, 2>();
    deallog.pop();
  }

  Kokkos::finalize();

  return 0;
}

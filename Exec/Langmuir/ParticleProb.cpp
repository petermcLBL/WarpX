
//
// Each problem must have its own version of PhysicalParticleContainer::InitData()
// to initialize the particle data.  It must also initialize charge and mass.
//

#include <cmath>

#include <BLProfiler.H>

#include <ParticleContainer.H>
#include <WarpXConst.H>

void
PhysicalParticleContainer::InitData()
{
    BL_PROFILE("PPC::InitData()");

    if (species_id == 0) {
	charge = -PhysConst::q_e;
	mass = PhysConst::m_e;
    } else if (species_id == 1) {
	charge = PhysConst::q_e;
	mass = PhysConst::m_p;	
    } else {
	BoxLib::Abort("PhysicalParticleContainer::InitData(): species_id must be 0 or 1");
    }

    m_particles.resize(GDB().finestLevel()+1);

    const int lev = 0;

    const Geometry& geom = GDB().Geom(lev);
    const Real* dx  = geom.CellSize();

    Real weight, ux, uy, uz;
    Real particle_xmin, particle_xmax, particle_ymin, particle_ymax, particle_zmin, particle_zmax;
    int n_part_per_cell;
    {
      ParmParse pp("langmuirwave");
      n_part_per_cell = 1;
      pp.query("num_particles_per_cell", n_part_per_cell);
      weight = 1.e25;
      pp.query("n_e", weight);
      #if BL_SPACEDIM==3
      weight *= dx[0]*dx[1]*dx[2]/n_part_per_cell;
      #elif BL_SPACEDIM==2
      weight *= dx[0]*dx[1]/n_part_per_cell;
      #endif

      particle_xmin = particle_ymin = particle_zmin = -2.e-5;
      particle_xmax = particle_ymax = particle_zmax =  2.e-5;
      pp.query("particle_xmin", particle_xmin);
      pp.query("particle_xmax", particle_xmax);
      pp.query("particle_ymin", particle_ymin);
      pp.query("particle_ymax", particle_ymax);
      pp.query("particle_zmin", particle_zmin);
      pp.query("particle_zmax", particle_zmax);   
 
      ux = 0.;
      uy = 0.;
      uz = 0.;
      if (species_id == 0) { // electrons
	  pp.query("ux", ux);
	  pp.query("uy", uy);
	  pp.query("uz", uz);
      }

      ux *= PhysConst::c;
      uy *= PhysConst::c;      
      uz *= PhysConst::c;
    }

    const BoxArray& ba = GDB().ParticleBoxArray(lev);
    const DistributionMapping& dm = GDB().ParticleDistributionMap(lev);

    MultiFab dummy_mf(ba, 1, 0, dm, Fab_noallocate);

    for (MFIter mfi(dummy_mf,false); mfi.isValid(); ++mfi)
    {
	int gid = mfi.index();
        Box grid = ba[gid];
        RealBox grid_box { grid,dx,geom.ProbLo() };

#if (BL_SPACEDIM == 3)
	int nx = grid.length(0), ny = grid.length(1), nz = grid.length(2);
#elif (BL_SPACEDIM == 2)
	int nx = grid.length(0), ny = 1, nz = grid.length(1);
#endif

	for (int k = 0; k < nz; k++) {
	  for (int j = 0; j < ny; j++) {
	    for (int i = 0; i < nx; i++) {
	      for (int i_part=0; i_part<n_part_per_cell;i_part++) {
		Real particle_shift = (0.5+i_part)/n_part_per_cell;
#if (BL_SPACEDIM == 3)
		Real x = grid_box.lo(0) + (i + particle_shift)*dx[0];
		Real y = grid_box.lo(1) + (j + particle_shift)*dx[1];
		Real z = grid_box.lo(2) + (k + particle_shift)*dx[2];
#elif (BL_SPACEDIM == 2)
		Real x = grid_box.lo(0) + (i + particle_shift)*dx[0];
		Real y = 0.0;
		Real z = grid_box.lo(1) + (k + particle_shift)*dx[1];
#endif   

		if (x >= particle_xmax || x < particle_xmin ||
		    y >= particle_ymax || y < particle_ymin ||
		    z >= particle_zmax || z < particle_zmin ) continue;
	      
		ParticleType p;

		p.m_id  = ParticleBase::NextID();
		p.m_cpu = ParallelDescriptor::MyProc();
		p.m_lev = lev;
		p.m_grid = gid; 

#if (BL_SPACEDIM == 3)
		p.m_pos[0] = x;
		p.m_pos[1] = y;
		p.m_pos[2] = z;
#elif (BL_SPACEDIM == 2)
		p.m_pos[0] = x;
		p.m_pos[1] = z;
#endif
		
		for (int i = 0; i < BL_SPACEDIM; i++) {
		  BL_ASSERT(p.m_pos[i] < grid_box.hi(i));
		}
		
		p.m_data[PIdx::w] = weight;
	      
		for (int i = 1; i < PIdx::nattribs; i++) {
		  p.m_data[i] = 0;
		}
	      
		p.m_data[PIdx::ux] = ux;
		p.m_data[PIdx::uy] = uy;
		p.m_data[PIdx::uz] = uz;

		if (!ParticleBase::Where(p,m_gdb)) // this will set m_cell
		{
		    BoxLib::Abort("invalid particle");
		}
		
		BL_ASSERT(p.m_lev >= 0 && p.m_lev <= GDB().finestLevel());
		//
		// Add it to the appropriate PBox at the appropriate level.
		//
		m_particles[p.m_lev][p.m_grid].push_back(p);
	      }
	    }
	  } 
        }
    }

    //
    // We still need to redistribute in order to define each particle's cell, grid and level, but this 
    //    shouldn't require any inter-node communication because the particles should already be in the right grid.
    //
    Redistribute(true);
}

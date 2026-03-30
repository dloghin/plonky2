mod allocator;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use plonky2::field::types::Field;
use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::prover::{
    compute_quotient_polys_for_bench, prepare_quotient_polys_bench_inputs,
};
use plonky2::util::timing::TimingTree;
#[cfg(feature = "cuda")]
use zeknox::{init_cuda_degree_rs, init_cuda_rs};

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

fn build_workload_circuit(
    num_steps: usize,
) -> (
    plonky2::plonk::circuit_data::CircuitData<F, C, D>,
    PartialWitness<F>,
) {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let a = builder.add_virtual_target();
    let b = builder.add_virtual_target();

    let mut prev = a;
    let mut cur = b;
    for _ in 0..num_steps {
        let t = builder.add(prev, cur);
        prev = cur;
        cur = t;
    }

    builder.register_public_input(a);
    builder.register_public_input(b);
    builder.register_public_input(cur);

    let mut pw = PartialWitness::new();
    pw.set_target(a, F::from_canonical_u64(3));
    pw.set_target(b, F::from_canonical_u64(5));

    (builder.build::<C>(), pw)
}

fn bench_compute_quotient_polys(c: &mut Criterion) {
    #[cfg(feature = "cuda")]
    init_cuda_rs();
    #[cfg(feature = "cuda")]
    init_cuda_degree_rs(18);

    let mut group = c.benchmark_group("compute_quotient_polys");
    group.sample_size(10);

    for steps in [1 << 10, 1 << 12, 1 << 13] {
        let (data, pw) = build_workload_circuit(steps);
        let mut timing = TimingTree::new("quotient-bench-setup", log::Level::Error);
        let prepared = prepare_quotient_polys_bench_inputs::<C, D>(
            &data.prover_only,
            &data.common,
            generate_partial_witness::<F, C, D>(pw, &data.prover_only, &data.common),
            &mut timing,
        );

        group.bench_with_input(BenchmarkId::from_parameter(steps), &steps, |b, _| {
            b.iter(|| {
                let polys = compute_quotient_polys_for_bench::<C, D>(
                    &data.common,
                    &data.prover_only,
                    &prepared,
                )
                .expect("compute_quotient_polys_for_bench");
                black_box(polys);
            });
        });
    }
}

criterion_group!(benches, bench_compute_quotient_polys);
criterion_main!(benches);
